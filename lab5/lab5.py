# %%
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
from torch import Tensor
from torch.distributions import MultivariateNormal as MVN

torch.set_default_device("cuda")
print(DEVICE := torch.get_default_device())


def make_batch(X: np.ndarray, batch_size: int):
    samples = np.random.choice(X.shape[0], (X.shape[0] // batch_size, batch_size), replace=False)
    for sample in samples:
        yield torch.tensor(X[sample, ...]).float()


def show(X: Tensor, title: str = "", save_path: str | None = None):
    X = X.cpu().numpy()
    plt.scatter(X[:, 0], X[:, 1], marker=".", c="k", s=0.5)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


# %%
# Load data
X = np.loadtxt("./bicycle.txt", dtype=np.float32)

# Plot a subset of data
show(next(make_batch(X, batch_size=30_000)))

# %%
# -----------------------------------------------
# Experiment with different variance schedules
# -----------------------------------------------

T = 1_000  # Max number of diffusion steps

# 1. Assume that β(t) = β_i + t * (β_f - β_i) / (T - 1) and compute sequences a(t), ā(t)
#    using a(t) := 1 - β(t) and ā(t) := Π_{j=0}^{t} a(j)
β_i, β_f = 1e-4, 2e-2
β = β_i + np.arange(T) * (β_f - β_i) / (T - 1)
a = 1 - β
ā = np.cumprod(a)

# fmt:off
plt.plot(ā, c="r", ls="-", label=r"$\overline{\alpha}_t = \prod_{j=0}^t (1 - \beta_j)$"+"\n"+r"for $\beta_t = \beta_i + t (\beta_f - \beta_i)/(T-1)$")
# fmt:on

# 2. Assume that ā(t) = σ(k - 2*k*t/T), where σ is the standard logistic function and compute
#    sequences a(t) and β(t) using a(t) = ā(t) / ā(t-1) and β(t) = 1 - a(t)
σ = lambda x: 1 / (1 + np.exp(-x))
k = 10
ā = σ(k - 2 * k * np.arange(T) / T)
a = np.concat(([ā[0]], ā[1:] / ā[:-1]))
β = 1 - a

assert np.allclose(np.cumprod(a), ā)

plt.plot(ā, c="b", ls="-", label=r"$\overline{\alpha}_t = \sigma\left(k - 2kt/T\right)$ for $k=$" + f"{k}")

# Plot schedules
plt.vlines(T / 2, 0.0, 1.0, ls="--", colors="k")
plt.xlabel(r"$t$")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# To Tensor
ā = torch.tensor(ā).float()
a = torch.tensor(a).float()
β = torch.tensor(β).float()

# %%
# Prepare directories
os.makedirs("./forward", exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs("./chckpts", exist_ok=True)


# %%
# -------------------------------------
# Define forward diffusion process
# -------------------------------------


def diffuse(x: Tensor, t: int, β: Tensor) -> Tensor:
    assert 0 <= t < β.shape[0]
    ā = torch.cumprod(1 - β, dim=0)
    return MVN(torch.sqrt(ā[t]) * x, (1 - ā[t]) * torch.eye(2)).sample()


x_batch = next(make_batch(X, batch_size=20_000))
for t in (0, 5, 10, 50, 100, 250, 500, 999):
    show(diffuse(x_batch, t, β), title=f"Step {t}", save_path=f"./forward/step{t}.png")


# %%
# ---------------------------
# Define DDPM model
# ---------------------------


class LearnableSinusoidalEmbedding(nn.Module):

    def __init__(self, output_dim: int = 128, latent_dim: int = 50, max_period: int = 10_000):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, output_dim)
        self.lin2 = nn.Linear(output_dim, output_dim)
        self.pe = lambda t: self.sinusoidal_positional_encoding(t, dim=latent_dim, max_period=max_period)

    def sinusoidal_positional_encoding(self, t: Tensor | int, dim: int, max_period: int) -> Tensor:
        assert dim % 2 == 0
        if type(t) is int:
            t = torch.tensor([[t]])

        freqs = 1 / torch.pow(max_period, 2 / dim * torch.arange(0, dim // 2))

        embeddings = torch.zeros(t.shape[0], dim)
        embeddings[:, 0::2] = torch.sin(freqs * t)
        embeddings[:, 1::2] = torch.cos(freqs * t)

        return embeddings

    def forward(self, t: Tensor | int) -> Tensor:
        return self.lin2(F.relu(self.lin1(self.pe(t))))


class DDPM(nn.Module):

    def __init__(self, dim: int = 2, latent_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.emb = LearnableSinusoidalEmbedding(output_dim=latent_dim)
        self.lin1 = nn.Linear(dim, latent_dim)
        self.lin2 = nn.Linear(latent_dim, latent_dim)
        self.lin3 = nn.Linear(latent_dim, latent_dim)
        self.lin4 = nn.Linear(latent_dim, dim)

    def forward(self, x: Tensor, t: Tensor | int) -> Tensor:
        t = self.emb(t)
        x = F.relu(t + self.lin1(x))
        x = F.relu(t + self.lin2(x))
        x = F.relu(t + self.lin3(x))
        x = self.lin4(x)
        return x

    @torch.no_grad()
    def sample(self, size: int, T: int, β: Tensor, x0: Tensor | None = None) -> Tensor:
        assert x0 is None or x0.shape == (size, self.dim)
        assert β.shape[0] >= T

        self.eval()

        mvn = MVN(loc=torch.zeros(self.dim), covariance_matrix=torch.eye(self.dim))
        a = 1 - β
        ā = torch.cumprod(a, dim=0)
        x = mvn.sample((size,)) if x0 is None else x0

        for t in trange(T - 1, -1, -1):
            z = mvn.sample((size,)) if t > 0 else torch.zeros((size, self.dim))
            ϵ_θ = self(x, t)
            x = 1 / torch.sqrt(a[t]) * (x - (1 - a[t]) / torch.sqrt(1 - ā[t]) * ϵ_θ) + torch.sqrt(β[t]) * z

        return x


# %%
# ---------------------------
# Define training parameters
# ---------------------------

epochs = 1000
log_period = 10
batch_size = 64

ddpm = DDPM(latent_dim=128)
criterion = F.mse_loss
optimizer = optim.Adam(ddpm.parameters(), lr=1e-4)

mvn = MVN(loc=torch.zeros(2), covariance_matrix=torch.eye(2))
sample_size = 10_000
x0 = mvn.sample((sample_size,))
# %%
# ---------------------------
# Training loop
# ---------------------------

loss_hist = []

for epoch in (pbar := trange(epochs)):
    ddpm.train()
    running_loss = 0.0

    for x_batch in make_batch(X, batch_size):
        optimizer.zero_grad()

        t = torch.tensor(np.random.choice(T, size=(batch_size, 1)))
        ϵ = mvn.sample((batch_size,))
        ϵ_θ = ddpm(torch.sqrt(ā[t]) * x_batch + torch.sqrt(1 - ā[t]) * ϵ, t)

        loss = criterion(ϵ_θ, ϵ)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = (running_loss * batch_size) / len(X)
    loss_hist.append(epoch_loss)
    pbar.set_description(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f}")

    if epoch == 0 or (epoch + 1) % log_period == 0:
        show(ddpm.sample(sample_size, T, β, x0), save_path=f"./results/r{epoch+1}.png")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": ddpm.state_dict(),
                "optim_state": optimizer.state_dict(),
                "loss": loss_hist,
            },
            f"./chckpts/chk{epoch+1}.pt",
        )

# %%
