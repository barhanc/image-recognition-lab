# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from PIL import Image
from tqdm import trange
from common import *

# %%
with open("checkpoints/loss_epoch2000.pickle", "rb") as file:
    loss_hist = pickle.load(file)

plt.plot(loss_hist["G"], label="Generator loss")
plt.plot(loss_hist["D"], label="Discriminator loss")
plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.legend(loc="best")
plt.show()


# %%
img = np.array(Image.open("crawled_cakes/027_99511daa.jpg").convert("RGB").resize((64, 64)))
img = img.transpose(2, 0, 1)
img = img.astype(np.float32)
img = (img / 127.5) - 1.0
img = Tensor(img).unsqueeze(0).float().to(DEVICE)
show(img)

# %%

os.makedirs("./noise_out/", exist_ok=True)

epochs = 20_000
log_period = 5000
params = torch.load("checkpoints/params_epoch500.pt", weights_only=False)

G = Generator()
G.load_state_dict(params["G_state_dict"])

noise = torch.randn(1, 128)
noise.requires_grad = True

criterion = F.mse_loss

optimizer = optim.SGD([noise], lr=0.5, momentum=0.99, nesterov=True)


for epoch in (pbar := trange(epochs)):
    optimizer.zero_grad()
    fake = G(noise)
    loss = criterion(fake, img)
    loss.backward()
    optimizer.step()
    pbar.set_description(f"Loss {loss.item():.4f}")
    if epoch == 0 or (epoch + 1) % log_period == 0:
        show(fake.detach(), savepath=f"./noise_out/out_{epoch+1}.png")


# %%

torch.save(noise, "./noise_out/noise.pt")

# %%
show(img)
show(G(noise).detach())

# %%
