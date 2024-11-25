# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import trange


torch.set_default_device("cuda")
print(DEVICE := torch.get_default_device())

# %%

MEAN = np.array([0.49139968, 0.48215827, 0.44653124])
STD = np.array([0.24703233, 0.24348505, 0.26158768])
transforms = v2.Compose(
    [
        v2.ToTensor(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        v2.Normalize(mean=MEAN, std=STD),
    ]
)


TRAIN_SET = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms)
TRAIN_SET, VALID_SET = torch.utils.data.random_split(TRAIN_SET, [0.9, 0.1], generator=torch.Generator(device="cuda"))
TEST_SET = datasets.CIFAR10(root="./data", train=False, download=True, transform=v2.ToTensor())

labels_map = {
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

# Plot some images with labels
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(TRAIN_SET), size=(1,)).item()
    img, label = TRAIN_SET[sample_idx]
    img = v2.Normalize(mean=-MEAN / STD, std=1 / STD)(img)
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0).squeeze(), cmap="gray")
plt.show()

# %%


def get_patches(X: Tensor, patch_size: int) -> Tensor: ...


# %%

batch_size = 64
loaders = {
    "train": DataLoader(TRAIN_SET, batch_size=batch_size, shuffle=True),
    "valid": DataLoader(VALID_SET, batch_size=batch_size, shuffle=True),
    "test": DataLoader(TEST_SET, batch_size=batch_size, shuffle=True),
}
