# %%
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tinygrad import Tensor, Device

# Make sure we are using GPU
print(Device.DEFAULT)


def show(img: Tensor, title=None):
    plt.imshow(img[0, 0, ...].numpy() if isinstance(img, Tensor) else img, cmap="grey")
    plt.axis("off")
    plt.title(title) if title is not None else ...
    plt.tight_layout()
    plt.show()


# Load an image of your choice
img_path = "./img.jpg"
img = np.asarray(Image.open(img_path).convert("RGB"))

show(img)

# Convert img to 3-D tensor and make sure that the values are in the range [0,1]
img = Tensor(img, requires_grad=False)
img = img / img.max()

# First dimension should be the channels
img = img.permute(order=(2, 0, 1))

# Add batch dimension
img = img.unsqueeze(0)

# Convert image to greyscale using a 1x3x1x1 convolution map. We use the NTSC formula
# Y = 0.299*r + 0.587*g + 0.114*b
r_coef, g_coef, b_coef = 0.299, 0.587, 0.114
filter_gray = Tensor([[[r_coef]], [[g_coef]], [[b_coef]]]).unsqueeze(0)
img = img.conv2d(filter_gray, padding=0, stride=1)

show(img)

# We use pooling to slightly decrease the image size
img_avgpool = img.avg_pool2d(kernel_size=(4, 4))
img_maxpool = img.max_pool2d(kernel_size=(4, 4))

show(img_avgpool, title="Average pooling")

show(img_maxpool, title="Max pooling")

# Based on the results we choose average pooling as it makes the image less pixelated
img = img_avgpool

# %%
# We now use gaussian bluring to get rid of noise and unimportant details


def filter_gauss(n: int) -> Tensor:
    phi = lambda mx, my, sigma: lambda x, y: Tensor.exp(-1 / (2 * sigma**2) * ((x - mx) ** 2 + (y - my) ** 2))
    x = Tensor.arange(n).repeat(n, 1)
    y = x.T
    f = phi(mx=(n - 1) / 2, my=(n - 1) / 2, sigma=n / 4)(x, y)
    f /= f.sum()
    return f


for n in (2, 3, 4, 5):
    # NOTE: for n%2==1 the output image has dimensions (W-1,H-1)
    show(img.conv2d(filter_gauss(n).reshape(1, 1, n, n), stride=1, padding=n // 2), title=f"Gaussian blur n={n}")

# Based on the results we choose a 3x3 gaussian kernel for bluring
n = 3
img = img.conv2d(filter_gauss(n).reshape(1, 1, n, n), stride=1, padding=n // 2)

# %%

Kx = Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float").reshape(1, 1, 3, 3)
Ky = Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype="float").reshape(1, 1, 3, 3)

Ix = img.conv2d(Kx, stride=1, padding=1)
Iy = img.conv2d(Ky, stride=1, padding=1)

G_amp = Tensor.sqrt(Ix**2 + Iy**2)
G_tan = Iy / Ix

show(G_amp, title="Gradient amplitude")

# %%
