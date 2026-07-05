import matplotlib.pyplot as plt

from src.datasets.aria_dataset import ARIADataset
from src.preprocess.fft import fft_magnitude, gaussian_residual
from src.preprocess.dct import dct_transform
from src.preprocess.residual import residual_transform

def show(x):
    x = x.detach().cpu().squeeze().numpy()
    plt.imshow(x, cmap="inferno")
    plt.axis("off")

ds = ARIADataset("data/val")
sample = ds[0]

img = sample["img"].unsqueeze(0)
img_res = gaussian_residual(img)
fft = fft_magnitude(gaussian_residual((img)))
dct = dct_transform(img)
residual = residual_transform(img)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0][0].imshow(img_res.squeeze().cpu().numpy())
ax[0][0].set_title("IMG")

plt.sca(ax[0][1])
show(fft)
ax[0][1].set_title("FFT")

plt.sca(ax[1][0])
show(dct)
ax[1][0].set_title("DCT")

plt.sca(ax[1][1])
show(residual)
ax[1][1].set_title("Residual")

for a in ax.ravel():
    a.axis("off")

plt.tight_layout()
plt.show()