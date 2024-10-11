# %%
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

img = np.asarray(Image.open("./img.jpg"))
plt.imshow(img)

from tinygrad import Tensor

img = Tensor(img, requires_grad=False) / 255
print(img.shape, img.dtype)

# %%
coeffR = 0.2126
coeffG = 0.7152
coeffB = 0.0722
img_greyscale = coeffR * img[:, :, 0] + coeffG * img[:, :, 1] + coeffB * img[:, :, 1]

print(img_greyscale.shape)
plt.imshow(img_greyscale.numpy(), cmap="grey")

# %%
