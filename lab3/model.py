import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

torch.set_default_device("cuda")
DEVICE = torch.get_default_device()


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.lin = nn.Linear(64 * 8 * 8, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = x.flatten(1)
        x = self.lin(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.lin = nn.Linear(latent_dim, 64 * 8 * 8)
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 3, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = x.reshape(-1, 64, 8, 8)
        x = F.leaky_relu(self.bn1(self.conv1(F.interpolate(x, scale_factor=2))), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(F.interpolate(x, scale_factor=2))), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(F.interpolate(x, scale_factor=2))), negative_slope=0.2)
        x = F.tanh(self.conv4(x))
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
