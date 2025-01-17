# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from copy import deepcopy
from random import choices

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
from torchview import draw_graph
from torch import Tensor
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2


torch.set_default_device("cuda")
print(DEVICE := torch.get_default_device())

# %%
# Create data augmentation and normalization pipeline
# ----------------------------------------------------

CIFAR_MEAN = np.array([0.49139968, 0.48215827, 0.44653124])
CIFAR_STD = np.array([0.24703233, 0.24348505, 0.26158768])

ToTensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
transforms = v2.Compose(
    [
        ToTensor,
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        v2.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)

# Download CIFAR-10

TRAIN_SET = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms)
TRAIN_SET, VALID_SET = torch.utils.data.random_split(TRAIN_SET, [0.9, 0.1], generator=torch.Generator(DEVICE))

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
# -----------------------------------------------


def show(imgs: Tensor, labels: dict[str, list] | None = None, figsize: tuple[int, int] = (5, 5)):
    imgs.unsqueeze(0) if len(imgs.shape) == 3 else ...

    B = imgs.shape[0]
    ncols = int(np.sqrt(B))
    nrows = int(np.ceil(B / ncols))
    fig = plt.figure(figsize=figsize)

    for b in range(B):
        fig.add_subplot(nrows, ncols, b + 1)
        if labels is not None:
            plt.title(
                f"True: {LABELS_MAP[labels['true'][b]]}\nPredicted: {LABELS_MAP[labels['pred'][b]]}",
                c="g" if labels["true"][b] == labels["pred"][b] else "r",
            )
        plt.axis("off")
        plt.imshow(imgs[b, ...].permute(1, 2, 0).cpu().numpy())
    plt.show()
    plt.close()


def show_patches(patches: Tensor, image_size: tuple[int, int], figsize: tuple[int, int] = (5, 5)):
    B = patches.shape[0]
    image_height, image_width = image_size
    patch_height, patch_width = patches.shape[3:]

    n_patches_x = image_width // patch_width
    n_patches_y = image_height // patch_height

    ncols = int(np.sqrt(B))
    nrows = int(np.ceil(B / ncols))
    fig = plt.figure(figsize=figsize)
    outer_gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    for col in range(ncols):
        for row in range(nrows):
            inner_gs = gridspec.GridSpecFromSubplotSpec(n_patches_y, n_patches_x, subplot_spec=outer_gs[row, col])

            for i in range(n_patches_x):
                for j in range(n_patches_y):
                    ax = fig.add_subplot(inner_gs[i, j])
                    ax.set_axis_off()
                    ax.imshow(patches[row * ncols + col, i * n_patches_y + j, ...].permute(1, 2, 0).cpu().numpy())

    plt.show()
    plt.close()


# %%
# Define module which divides the image into patches
# ----------------------------------------------------


# TODO: try to make this faster
class Patchify(nn.Module):
    def __init__(self, image_size: tuple[int, int], patch_size: tuple[int, int], flat: bool):
        super().__init__()
        self.image_height, self.image_width = image_size
        self.patch_height, self.patch_width = patch_size
        self.flatten = nn.Flatten(start_dim=2) if flat else nn.Identity()
        assert self.image_height % self.patch_height == 0
        assert self.image_width % self.patch_width == 0

    def forward(self, x: Tensor) -> Tensor:
        patches = torch.stack(
            [
                x[..., i : i + self.patch_height, j : j + self.patch_width]
                for i in range(0, self.image_height, self.patch_height)
                for j in range(0, self.image_width, self.patch_width)
            ],
            dim=1,
        )
        return self.flatten(patches)


# Check results on random images

imgs, _ = zip(*choices(TRAIN_SET, k=9))
imgs = torch.stack(imgs)
imgs = v2.Normalize(mean=-CIFAR_MEAN / CIFAR_STD, std=1 / CIFAR_STD)(imgs)

show(imgs, figsize=(10, 10))
show_patches(Patchify(image_size=(32, 32), patch_size=(4, 4), flat=False)(imgs), image_size=(32, 32), figsize=(10, 10))


# %%
# Define ViT model
# -----------------------------------------------


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dim_head, dropout):
        super().__init__()

        embed_dim = dim_head * n_heads
        self.attn_output_weights: Tensor | None = None

        self.q = nn.Linear(dim, embed_dim, bias=False)
        self.k = nn.Linear(dim, embed_dim, bias=False)
        self.v = nn.Linear(dim, embed_dim, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout, bias=False, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(embed_dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        x, self.attn_output_weights = self.attn(q, k, v)
        return self.proj(x)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, dim_head: int, mlp_dim: int, dropout: float):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.atten_blk = MultiheadSelfAttention(dim, n_heads, dim_head, dropout)
        self.dense_blk = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.atten_blk(self.norm1(x))
        x = x + self.dense_blk(self.norm2(x))
        return x


class ViT(nn.Module):

    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: tuple[int, int],
        channels: int,
        n_classes: int,
        n_heads: int,
        dim: int,
        dim_mlp: int,
        dim_head: int,
        depth: int,
        dropout: float,
    ):
        super().__init__()

        image_height, image_width = image_size
        patch_height, patch_width = patch_size
        n_patches = (image_height // patch_height) * (image_width // patch_width)

        self.to_patch_embedding = nn.Sequential(
            Patchify(image_size, patch_size, flat=True),
            nn.Linear(channels * patch_height * patch_width, dim),
        )

        self.transformer_layers = [TransformerLayer(dim, n_heads, dim_head, dim_mlp, dropout) for _ in range(depth)]
        self.transformer = nn.Sequential(*self.transformer_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim))
        self.clf_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # (B, C, H, W)
        b = x.shape[0]

        # (B, T, N_emb)
        x = self.to_patch_embedding(x)

        # (B, T+1, N_emb)
        x = torch.cat((self.cls_token.repeat(b, 1, 1), x), dim=1)
        x = self.dropout(x + self.pos_embed)
        x = self.transformer(x)

        # (B, N_emb)
        x = x[:, 0, :]

        # (B, N_cls)
        x = self.clf_head(x)

        return x


# Plot keras-style architecture graph

vit = ViT(
    image_size=(32, 32),
    patch_size=(4, 4),
    channels=3,
    n_classes=10,
    n_heads=8,
    dim=256,
    dim_mlp=512,
    dim_head=32,
    depth=6,
    dropout=0.2,
)

model_graph = draw_graph(vit, input_size=(64, 3, 32, 32), depth=3, expand_nested=True, roll=True)
model_graph.visual_graph


# %%
# Define training parameters
# ---------------------------

batch_size = 64
dataloader = {
    "train": DataLoader(TRAIN_SET, batch_size=batch_size, shuffle=True, generator=torch.Generator(DEVICE)),
    "valid": DataLoader(VALID_SET, batch_size=batch_size, shuffle=True, generator=torch.Generator(DEVICE)),
}

epochs = 160
criterion = F.cross_entropy
optimizer = optim.AdamW(vit.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

# %%
# Training loop
# ---------------------------

acc_hist = {"train": [], "valid": []}

best_acc = 0.0
best_vit_params = deepcopy(vit.state_dict())

for epoch in (pbar := trange(epochs)):
    for phase in ["train", "valid"]:
        if phase == "train":
            vit.train()
        elif phase == "valid":
            vit.eval()

        running_hits = 0
        running_total = 0

        for X_batch, t_batch in dataloader[phase]:

            X_batch: Tensor = X_batch.to(DEVICE)
            t_batch: Tensor = t_batch.to(DEVICE)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                # Forward
                y_batch = vit(X_batch)
                loss = criterion(y_batch, t_batch)

                # Backward
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # Update running statistics
                running_hits += (torch.argmax(y_batch, 1) == t_batch).sum().item()
                running_total += t_batch.size(0)

        # Compute epoch statistics
        epoch_acc = running_hits / running_total

        # Save epoch statistics
        acc_hist[phase].append(epoch_acc)

        # Update best model
        if phase == "valid" and epoch_acc > best_acc:
            best_acc, best_vit_params = epoch_acc, deepcopy(vit.state_dict())

    pbar.set_description(
        f"Epoch [{epoch+1}/{epochs}]| "
        f"b.acc={best_acc*100:.2f}% | "
        f"v.acc={acc_hist['valid'][-1]*100:.2f}% | "
        f"t.acc={acc_hist['train'][-1]*100:.2f}% | "
    )

# Save best model
torch.save({"model_state": best_vit_params, "acc_hist": acc_hist}, "./chk.pt")

# %%
# Plot accuracy history

acc_hist = torch.load("./extra/chk.pt", weights_only=False)["acc_hist"]

plt.plot(acc_hist["train"], label="Train")
plt.plot(acc_hist["valid"], label="Valid")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()
plt.close()


# %%
# Evaluate model on test set
# ---------------------------

TEST_SET = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=v2.Compose([ToTensor, v2.Normalize(CIFAR_MEAN, CIFAR_STD)]),
)
TEST_LOADER = DataLoader(TEST_SET, batch_size=batch_size, shuffle=True, generator=torch.Generator(DEVICE))

vit.load_state_dict(torch.load("./extra/chk.pt", weights_only=False)["model_state"])
vit.eval()

hits, total = 0, 0

with torch.no_grad():
    for X_batch, t_batch in TEST_LOADER:
        X_batch = X_batch.to(DEVICE)
        t_batch = t_batch.to(DEVICE)
        y_batch = vit(X_batch)
        hits += (torch.argmax(y_batch, 1) == t_batch).sum().item()
        total += t_batch.size(0)

print(f"Test accuracy = {hits/total*100:.2f}%")

# %%
# Plot some predictions
# ----------------------

test_imgs, test_labels = zip(*choices(TEST_SET, k=9))
test_imgs, test_labels = torch.stack(test_imgs), torch.tensor(test_labels)

pred_labels: Tensor = vit(test_imgs.to(DEVICE))
pred_labels = pred_labels.detach().argmax(1).cpu()

test_imgs = v2.Normalize(mean=-CIFAR_MEAN / CIFAR_STD, std=1 / CIFAR_STD)(test_imgs)
show(test_imgs, labels={"true": test_labels.tolist(), "pred": pred_labels.tolist()}, figsize=(10, 10))


# %%
# Visualize attention using attention rollout
# --------------------------------------------


def attention_rollout(As: list[Tensor], i: int = 0) -> Tensor:
    assert 0 <= i < len(As)
    R = As[i]
    for A in As[i:]:
        R = R @ (A + torch.eye(A.shape[1]))
    return R


As = [layer.atten_blk.attn_output_weights.detach() for layer in vit.transformer_layers]
R = attention_rollout(As)
R = R[:, 0, 1:].reshape(-1, 1, 8, 8)
R = F.interpolate(R, (32, 32), mode="bicubic")

m, _ = R.min(0)
M, _ = R.max(0)
R = (R - m) / (M - m)
R = R.repeat((1, 3, 1, 1), R)

imgs_with_attn = R * test_imgs
show(imgs_with_attn, figsize=(10, 10))

# %%
