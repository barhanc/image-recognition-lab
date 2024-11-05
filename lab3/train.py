# %%
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import trange

import torch
import torch.optim as optim

from torch import Tensor

from lab3.model import *

# %%
dirname = "./crawled_cakes"
imgs_dataset = np.array([Image.open(f"{dirname}/{f}").convert("RGB").resize((64, 64)) for f in os.listdir(dirname)])
imgs_dataset = imgs_dataset.transpose(0, 3, 1, 2)
imgs_dataset = imgs_dataset.astype(np.float32)
imgs_dataset = (imgs_dataset / 127.5) - 1.0

# show(next(make_batch(imgs_dataset, 36)))

# %%
batch_size = 16
log_period = 50
epochs = 3_000

G = Generator(latent_dim := 128)
D = Discriminator()

criterion = F.binary_cross_entropy_with_logits

optim_g = optim.AdamW(G.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-2)
optim_d = optim.AdamW(D.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-2)

fixed_noise = torch.randn(36, latent_dim)
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
        noise = torch.randn(batch_size, latent_dim)
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
                "optim_g_state": optim_g.state_dict(),
                "optim_d_state": optim_d.state_dict(),
            },
            f"./checkpoints/params_epoch{epoch+1}.pt",
        )
        with open(f"./checkpoints/loss_epoch{epoch+1}.pickle", "wb") as file:
            pickle.dump(hist_loss, file, protocol=pickle.HIGHEST_PROTOCOL)

# Plot loss history
plt.plot(hist_loss["G"], label="Generator loss")
plt.plot(hist_loss["D"], label="Discriminator loss")
plt.legend(loc="best")
plt.savefig("./loss_hist.png", bbox_inches="tight")
plt.close()

# %%
