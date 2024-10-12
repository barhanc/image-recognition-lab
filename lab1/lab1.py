# %%
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tinygrad import Tensor, Device

# Make sure we are using GPU
print(Device.DEFAULT)


# Define simple helper function for easy results plotting
def show(img: Tensor, title=None):
    plt.imshow(img[0, ...].permute(1, 2, 0).numpy() if isinstance(img, Tensor) else img, cmap="grey")
    plt.axis("off")
    plt.title(title) if title is not None else ...
    plt.tight_layout()
    plt.show()


# Load an image of your choice
img_path = "./img.jpg"
img = np.array(Image.open(img_path).convert("RGB"))

show(img)

# %%
# Convert img to 3-D tensor and make sure that the values are in the range [0,1]
img = Tensor(img, requires_grad=False)
img = img / img.max()

# First dimension should be the channels
img = img.permute(order=(2, 0, 1))

# Add batch dimension
img = img.unsqueeze(0)

# Save original image for later
img_og = img

# Convert image to greyscale using a 1x3x1x1 convolution map. We use the NTSC formula
# Y = 0.299*r + 0.587*g + 0.114*b
r_coef, g_coef, b_coef = 0.299, 0.587, 0.114
filter_gray = Tensor([[[r_coef]], [[g_coef]], [[b_coef]]]).unsqueeze(0)
img = img.conv2d(filter_gray, padding=0, stride=1)

show(img)

# %%
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
    phi = lambda mx, my, s: lambda x, y: Tensor.exp(-1 / (2 * s**2) * ((x - mx) ** 2 + (y - my) ** 2))
    x = Tensor.arange(n).repeat(n, 1)
    y = x.T
    f = phi(mx=(n - 1) / 2, my=(n - 1) / 2, s=n / 4)(x, y)
    f /= f.sum()
    return f


for n in (3, 5, 7):
    show(img.conv2d(filter_gauss(n).reshape(1, 1, n, n), stride=1, padding=n // 2), title=f"Gaussian blur n={n}")

# Based on the results above we choose a 3x3 gaussian kernel for bluring
n = 3
img = img.conv2d(filter_gauss(n).reshape(1, 1, n, n), stride=1, padding=n // 2)

# %%
# We now apply the Sobel filters Kx, Ky and compute gradient information -- amplitude and angle

Kx = Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float").reshape(1, 1, 3, 3)
Ky = Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype="float").reshape(1, 1, 3, 3)

Ix = img.conv2d(Kx, stride=1, padding=1)
Iy = img.conv2d(Ky, stride=1, padding=1)

G_amp = Tensor.sqrt(Ix**2 + Iy**2)
G_tan = Iy / Ix

show(G_amp, title="Gradient amplitude")

# %%
# Our previous result already looks pretty good, but we want the edges to be thinner. In order to do
# so we perform the non-max suppression

from math import tan, pi

# First we compute masks for every direction (E-W, N-S, NE-SW, NW-SE). Here we also note that
# directly computing angle by arctan is inefficient.
tan22_5 = tan(1 * pi / 8)
tan67_5 = tan(3 * pi / 8)

mask_00 = ((G_tan < +tan22_5) * (G_tan > -tan22_5)).flatten()
mask_90 = ((G_tan < -tan67_5) + (G_tan > +tan67_5)).flatten()
mask_p45 = ((G_tan >= +tan22_5) * (G_tan <= +tan67_5)).flatten()
mask_n45 = ((G_tan <= -tan22_5) * (G_tan >= -tan67_5)).flatten()

# After computing masks, we compute strides which determine the location of neighbors according to
# gradient direction. Note here that we use a flattened representation as it makes indexing easier.
*_, H, W = G_amp.shape
strides = 1 * mask_00 + W * mask_90 + (W + 1) * mask_p45 + (W - 1) * mask_n45

G_amp_flat = G_amp.flatten()
idxs = Tensor.arange(W * H)
mask = (G_amp_flat >= G_amp_flat[idxs + strides]) * (G_amp_flat >= G_amp_flat[idxs - strides])

# Now we only need to take care of border artifacts
mask = mask.reshape(1, 1, H, W)
mask = mask[..., 1 : H - 1, 1 : W - 1]
mask = mask.pad2d((1, 1, 1, 1))

G_amp *= mask

show(G_amp, title="Gradient amplitude after non-max suppression")

# %%
# The next thing to do is to pass the gradient amplitudes through a simple thresholded relu function
# and normalize the values so that they are either 1 (the pixel is part of some edge) or 0

for threshold in (0.3, 0.4, 0.5, 0.6):
    show((G_amp - threshold) > 0, title=f"Edges, threshold={threshold}")

# Based on the results above we choose a threshold of 0.4
threshold = 0.4
G_amp_threshold = (G_amp - threshold) > 0

# %%
# The last thing to do is to upscale the edges' channel and overlay it over the original image

edges = G_amp_threshold.conv_transpose2d(Tensor.ones(1, 1, 3, 3), stride=4, output_padding=(3, 1))

amplification = 0.8
r_channel = (img_og[:, 0, ...] - amplification * edges[:, 0, ...]).clip(0, 1)
g_channel = (img_og[:, 1, ...] + amplification * edges[:, 0, ...]).clip(0, 1)
b_channel = (img_og[:, 2, ...] - amplification * edges[:, 0, ...]).clip(0, 1)

img_with_edges = Tensor.stack(r_channel, g_channel, b_channel, dim=1)

show(img_with_edges)
