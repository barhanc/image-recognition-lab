# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils
import torch.utils.data
import torch.optim as optim

from copy import deepcopy

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# %%
# Make sure that we are utilizing a DL accelerator
DEVICE_CPU = torch.device("cpu")
DEVICE_GPU = torch.device("cuda")

device = DEVICE_GPU if torch.cuda.is_available() else DEVICE_CPU
print(device)

# %%
# Load CIFAR-10 dataset
train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=ToTensor())
test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=ToTensor())

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
    sample_idx = torch.randint(len(train_set), size=(1,)).item()
    img, label = train_set[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0).squeeze(), cmap="gray")
plt.show()

# %%
import torch.nn as nn
import torch.nn.functional as F


# TODO: Check how the initialization is handled in Pytorch
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), stride=1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(in_features=5 * 5 * 16, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, X):
        X = F.relu(self.avgpool1(self.conv1(X)))
        X = F.relu(self.avgpool2(self.conv2(X)))
        X = torch.flatten(X, 1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X


# %%


def train_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    epochs: int,
    optimizer_params: dict,
    device=DEVICE_GPU,
):
    optimizer = optim.Adam(params=model.parameters(), **optimizer_params)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    best_acc, best_model_wts = 0.0, deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch:4.0f}/{epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            print(f"{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc, best_model_wts = epoch_acc, deepcopy(model.state_dict())

        print()

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def eval_model(model: nn.Module, dataloader: DataLoader, device=DEVICE_GPU):
    model.to(device)
    model.eval()

    corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            corrects += torch.sum(preds == labels.data)

    test_acc = corrects / len(dataloader.dataset)
    print(f"Test Acc: {test_acc:.4f}")


# %%
train_set, val_set = torch.utils.data.random_split(train_set, [0.90, 0.10])

batch_size = 64
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

dataloaders = {"train": train_dataloader, "val": val_dataloader}

model = LeNet()

train_model(model, dataloaders, 100, optimizer_params={})
# %%
eval_model(model, test_dataloader)

# %%
