# %%
from ultralytics import YOLO

# Make sure that the model works correctly
YOLO("yolov8m.pt").predict(source="./catdog.jpg", save=True, conf=0.5)

# %%
import os
import pandas as pd

# Prepare a list of images to download and create directory structure for fine-tuning

os.system("wget https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv")
os.system("wget https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv")
os.system("wget https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv")

labels_map = pd.read_csv("oidv7-class-descriptions-boxable.csv")
labels = labels_map.loc[labels_map["DisplayName"].isin(["Food"]), "LabelName"]

df_train = pd.read_csv("oidv6-train-annotations-bbox.csv")
df_train = df_train.loc[df_train["LabelName"].isin(labels)]

df_valid = pd.read_csv("validation-annotations-bbox.csv")
df_valid = df_valid.loc[df_valid["LabelName"].isin(labels)]

with open("image_list_train.txt", "w") as f:
    f.write("")
    for i in df_train.index:
        f.write(f"train/{df_train.loc[i, 'ImageID']}\n")

with open("image_list_valid.txt", "w") as f:
    f.write("")
    for i in df_valid.index:
        f.write(f"validation/{df_valid.loc[i, 'ImageID']}\n")

for i in df_train.index:
    xmin, xmax = df_train.loc[i, "XMin"], df_train.loc[i, "XMax"]
    ymin, ymax = df_train.loc[i, "YMin"], df_train.loc[i, "YMax"]
    xc, yc, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
    with open(f'./datasets/oiv7/labels/train/{df_train.loc[i, "ImageID"]}.txt', "a") as f:
        f.write(f"0 {xc} {yc} {w} {h}\n")

for i in df_valid.index:
    xmin, xmax = df_valid.loc[i, "XMin"], df_valid.loc[i, "XMax"]
    ymin, ymax = df_valid.loc[i, "YMin"], df_valid.loc[i, "YMax"]
    xc, yc, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
    with open(f'./datasets/oiv7/labels/valid/{df_valid.loc[i, "ImageID"]}.txt', "a") as f:
        f.write(f"0 {xc} {yc} {w} {h}\n")

# %%
# Download images
# fmt:off
os.system('python3 downloader.py "image_list_train.txt" --download_folder="./datasets/oiv7/images/train/" --num_processes=5')
os.system('python3 downloader.py "image_list_valid.txt" --download_folder="./datasets/oiv7/images/valid/" --num_processes=5')
# fmt:on

# %%
import torch

# Fine-tune model for a few epochs on a custom dataset

# Encountered issue described in https://github.com/ultralytics/ultralytics/issues/1149 (possibly
# because of the GPU model -- GTX 1660 Super)
torch.backends.cudnn.enabled = False

model = YOLO("yolov8n.pt")
results = model.train(data="dataset.yaml", epochs=5, single_cls=True)

# %%
import cv2

# Test fine-tuned model on some test images from OIv7

TEST_DIR = "test_food"

for i in range(1, 5):
    results = model.predict(source=f"{TEST_DIR}/food{i}.jpg", save=True, conf=0.5)
    for r in results:
        img = r.orig_img
        for x, y, w, h in r.boxes.xywh.tolist():
            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
            blur_img = cv2.GaussianBlur(img[y : y + h, x : x + w], (101, 101), 0)
            img[y : y + h, x : x + w] = blur_img
            cv2.imwrite(f"test_food/food{i}_blur.jpg", img)
