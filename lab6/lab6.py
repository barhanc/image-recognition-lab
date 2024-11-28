# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
from torch import Tensor
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transform


torch.set_default_device("cuda")
print(DEVICE := torch.get_default_device())

# %%
# Create data augmentation and normalization pipeline

CIFAR_MEAN = np.array([0.49139968, 0.48215827, 0.44653124])
CIFAR_STD = np.array([0.24703233, 0.24348505, 0.26158768])

transforms = transform.Compose(
    [
        transform.ToTensor(),
        transform.RandomHorizontalFlip(p=0.5),
        transform.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transform.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)

TRAIN_SET = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms)
TRAIN_SET, VALID_SET = torch.utils.data.random_split(TRAIN_SET, [0.9, 0.1], generator=torch.Generator(DEVICE))
TEST_SET = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform.ToTensor())

LABELS_MAP = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

# %%
# Define helper functions for plotting


def show(imgs: Tensor, labels: Tensor | None = None):
    B = imgs.shape[0]

    ncols = int(np.sqrt(B))
    nrows = int(np.ceil(B / ncols))
    fig = plt.figure(figsize=(5, 5))

    for b in range(B):
        fig.add_subplot(nrows, ncols, b + 1)
        plt.axis("off")
        plt.imshow(imgs[b, ...].permute(1, 2, 0).cpu().numpy())
    plt.show()
    plt.close()


def show_patches(patches: Tensor):
    from math import sqrt

    fig = plt.figure(figsize=(5, 5))

    for i in range(T := patches.shape[1]):
        fig.add_subplot(int(sqrt(T)), int(sqrt(T)), i + 1)
        plt.axis("off")
        plt.imshow(patches[0, i, ...].permute(1, 2, 0), cmap="gray")
    plt.show()
    plt.close()


# %%
# Define function which divides the image into patches


def patchify(x: Tensor, sz: int) -> Tensor:
    assert x.shape[2] == x.shape[3]
    assert (n := x.shape[2]) % sz == 0
    return torch.stack([x[..., i : i + sz, j : j + sz] for i in range(0, n, sz) for j in range(0, n, sz)], 1)


# Check results on a random image from dataset
img, _ = TRAIN_SET[5]
img: Tensor = transform.Normalize(mean=-CIFAR_MEAN / CIFAR_STD, std=1 / CIFAR_STD)(img)
img = img.unsqueeze(0)

show(img)
show_patches(patchify(img, sz=4))

# %%
# Define ViT model


class TransformerBlock(nn.Module):

    def __init__(self, n_emb: int, n_heads: int, p: float):
        super().__init__()
        self.mattn_blk = nn.MultiheadAttention(n_emb, n_heads)
        self.dense_blk = nn.Sequential(
            nn.Linear(n_emb, 2 * n_emb),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(2 * n_emb, n_emb),
            nn.Dropout(p),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mattn_blk(F.layer_norm(x))
        x = x + self.dense_blk(F.layer_norm(x))
        return x


class ViT(nn.Module):

    def __init__(self, n_cls: int, n_emb: int, n_heads: int, n_blk: int, toks: int, patch_sz: int):
        super().__init__()
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(n_emb, n_heads, 0.2) for _ in range(n_blk)])
        self.lin1 = nn.Linear(patch_sz**2 * 3, n_emb)
        self.lin2 = nn.Linear(n_emb, n_cls)
        self.pe = torch.randn(toks + 1, n_emb, requires_grad=True)  # TODO: Fix
        self.sz = patch_sz

    def patchify(self, x: Tensor, sz: int) -> Tensor:
        assert x.shape[2] == x.shape[3]
        assert (n := x.shape[2]) % sz == 0
        return torch.stack([x[..., i : i + sz, j : j + sz] for i in range(0, n, sz) for j in range(0, n, sz)], 1)

    def forward(self, x: Tensor) -> Tensor:
        # (B, T, C, P, P)
        x = self.patchify(x, self.sz)

        # (B, T, N_emb)
        x = self.lin1(x.flatten(2))

        B, T, N_emb = x.shape

        # (B, T+1, N_emb)
        x = torch.concat([torch.zeros(B, N_emb), x], dim=1)
        x = F.dropout(x + self.pe, p=0.2)
        x = self.transformer_blocks(x)

        # (B, N_emb)
        x = F.layer_norm(x[:, 0, :])

        # (B, N_cls)
        x = self.lin2(x)

        return x


# %%

batch_size = 64
loaders = {
    "train": DataLoader(TRAIN_SET, batch_size=batch_size, shuffle=True),
    "valid": DataLoader(VALID_SET, batch_size=batch_size, shuffle=True),
    "test": DataLoader(TEST_SET, batch_size=batch_size, shuffle=True),
}
