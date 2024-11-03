# %%
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from tqdm import trange
from torch import Tensor

torch.set_default_device("cuda")
print(f"Default device: {(DEVICE := torch.get_default_device())}")


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.lin = nn.Linear(64 * 4 * 4, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = x.flatten(1)
        x = self.lin(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.lin = nn.Linear(latent_dim, 64 * 4 * 4)
        self.convt1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.convt2 = nn.ConvTranspose2d(64, 128, 4, 2, 1)
        self.convt3 = nn.ConvTranspose2d(128, 256, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv = nn.Conv2d(256, 3, 5, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = x.reshape(-1, 64, 4, 4)
        x = F.leaky_relu(self.bn1(self.convt1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.convt2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.convt3(x)), negative_slope=0.2)
        x = F.tanh(self.conv(x))
        return x


def show(imgs: Tensor, savepath: str | None = None):
    B, *_ = imgs.shape
    # Normalize values from [-1;+1] to [0;1]
    imgs = (imgs + 1.0) / 2.0

    ncols = int(np.sqrt(B))
    nrows = int(np.ceil(B / ncols))
    fig = plt.figure(figsize=(9, 9))

    for b in range(B):
        fig.add_subplot(nrows, ncols, b + 1)
        plt.axis("off")
        plt.imshow(imgs[b, ...].permute(1, 2, 0).cpu().numpy())

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def make_batch(imgs: np.ndarray, batch_size: int):
    samples = np.random.choice(len(imgs), (len(imgs) // batch_size, batch_size), replace=False)
    for sample in samples:
        yield Tensor(imgs[sample, ...]).float().to(DEVICE)


# %%
dirname = "./crawled_cakes"
imgs_dataset = np.array([Image.open(f"{dirname}/{f}").convert("RGB").resize((32, 32)) for f in os.listdir(dirname)])
imgs_dataset = imgs_dataset.transpose(0, 3, 1, 2)
imgs_dataset = imgs_dataset.astype(np.float32)
imgs_dataset = (imgs_dataset / 127.5) - 1.0

# show(next(make_batch(imgs_dataset, 36)))

# %%
batch_size = 16
log_period = 50
epochs = 3_000

G = Generator()
D = Discriminator()

criterion = F.binary_cross_entropy_with_logits

optim_g = optim.AdamW(G.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-2)
optim_d = optim.AdamW(D.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-2)

fixed_noise = torch.randn(36, 64)
# show(G(fixed_noise).detach())

# %%
hist_loss = {"G": [], "D": []}
os.makedirs("./outputs/", exist_ok=True)
os.makedirs("./checkpoints/", exist_ok=True)

# Training loop
for epoch in (pbar := trange(epochs)):
    running_loss_g = 0.0
    running_loss_d = 0.0

    for X_real in make_batch(imgs_dataset, batch_size):
        # TRAIN DISCRIMINATOR
        # =============================
        optim_d.zero_grad()

        # Train discriminator with real images
        t_real = torch.ones(batch_size, 1)
        y_real = D(X_real)

        # Train discriminator with fake images
        noise = torch.randn(batch_size, 64)
        X_fake: Tensor = G(noise)

        t_fake = torch.zeros(batch_size, 1)
        y_fake = D(X_fake.detach())

        loss_d = criterion(y_real, t_real) + criterion(y_fake, t_fake)
        loss_d.backward()

        optim_d.step()

        # TRAIN GENERATOR
        # =============================
        optim_g.zero_grad()

        t_adver = torch.ones(batch_size, 1)
        y_adver = D(X_fake)

        loss_g = criterion(y_adver, t_adver)
        loss_g.backward()

        optim_g.step()

        running_loss_g += loss_g.item()
        running_loss_d += loss_d.item()

    epoch_loss_g = (running_loss_g * batch_size) / len(imgs_dataset)
    epoch_loss_d = (running_loss_d * batch_size) / len(imgs_dataset)

    hist_loss["G"].append(epoch_loss_g)
    hist_loss["D"].append(epoch_loss_d)

    pbar.set_description(f"Epoch [{epoch+1}/{epochs}] | G Loss: {epoch_loss_g:.4f} | D Loss: {epoch_loss_d:.4f}")

    # Print losses
    if (epoch + 1) % log_period == 0:
        show(G(fixed_noise).detach(), savepath=f"./outputs/img{epoch+1}.png")
        torch.save(
            {
                "epoch": epoch + 1,
                "G_state_dict": G.state_dict(),
                "D_state_dict": D.state_dict(),
            },
            f"./checkpoints/params_epoch{epoch+1}.pt",
        )
        with open(f"./checkpoints/loss_epoch{epoch+1}.pickle", "wb") as file:
            pickle.dump(hist_loss, file, protocol=pickle.HIGHEST_PROTOCOL)

# Plot loss history
plt.plot(hist_loss["G"], label="Generator loss")
plt.plot(hist_loss["D"], label="Discriminator loss")
plt.savefig("./loss_hist.png", bbox_inches="tight")
plt.close()

# %%
