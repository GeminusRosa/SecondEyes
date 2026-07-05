import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import cast
from pathlib import Path
from torchvision import transforms

from src.preprocess.fft import fft_magnitude, gaussian_residual
from src.preprocess.dct import dct_transform
from src.preprocess.residual import residual_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([transforms.CenterCrop((256, 256)), transforms.ToTensor()])

def load_image(path: str | Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = cast(torch.Tensor, TRANSFORM(img))
    return tensor

def build_batch(paths) -> torch.Tensor:
    tensors = []
    for p in paths:
        tensors.append(load_image(p))
    return torch.stack(tensors, dim=0).to(DEVICE)

def normalize(x):
    x = x.astype(np.float32)
    x -= x.min()
    x /= x.max() + 1e-8
    return x

def process_image(img):
    with torch.no_grad():
        img_np = img.cpu().numpy()

        img_res = gaussian_residual(img)

        fft = fft_magnitude(img_res)
        dct = dct_transform(img)
        residual = residual_transform(img)

    return {
        "img": img_np,
        "img_res": img_res.cpu().numpy(),
        "fft": fft.cpu().numpy(),
        "dct": dct.cpu().numpy(),
        "residual": residual.cpu().numpy(),
    }

def build_fingerprint(img):
    processed = process_image(img)

    fft = processed["fft"][:, 0]
    dct = processed["dct"][:, 0]
    residual = processed["residual"][:, 0]

    return {
        "fft_mean": normalize(fft.mean(axis=0)),
        "fft_std": normalize(fft.std(axis=0)),
        "dct_mean": normalize(dct.mean(axis=0)),
        "dct_std": normalize(dct.std(axis=0)),
        "res_mean": normalize(residual.mean(axis=0)),
        "res_std": normalize(residual.std(axis=0)),
    }

def show(result, n):
    fig, ax = plt.subplots(1, 3, figsize=(16, 10))
    ax[0].imshow(result["fft_mean"], cmap="inferno")
    ax[0].set_title(f"FFT Mean ({n})", fontsize=16)

    ax[1].imshow(result["dct_mean"], cmap="inferno")
    ax[1].set_title(f"DCT Mean ({n})", fontsize=16)

    ax[2].imshow(result["res_mean"], cmap="inferno")
    ax[2].set_title(f"Residual Mean ({n})", fontsize=16)

    for a in ax.ravel():
        a.axis("off")

    plt.tight_layout()
    plt.show()

def collect_images(folder, count=100):
    folder = Path(folder)
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    files = []

    for ext in exts:
        files.extend(folder.glob(ext))

    files = sorted(files)

    if len(files) < count:
        raise ValueError(f"Only found {len(files)} images")

    return files[:count]

def preview_single(result):
    fig, ax = plt.subplots(1, 5, figsize=(18, 5))

    ax[0].imshow(normalize(result["img"][0].transpose(1, 2, 0)))
    ax[0].set_title("Original", fontsize=16)

    ax[1].imshow(normalize(result["img_res"][0, 0]), cmap="gray")
    ax[1].set_title("Gaussian Residual", fontsize=16)

    ax[2].imshow(normalize(result["fft"][0, 0]), cmap="inferno")
    ax[2].set_title("FFT", fontsize=16)

    ax[3].imshow(normalize(result["dct"][0, 0]), cmap="inferno")
    ax[3].set_title("DCT", fontsize=16)

    ax[4].imshow(normalize(result["residual"][0, 0]), cmap="inferno")
    ax[4].set_title("Residual", fontsize=16)

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    folder = "data/test/StarryAI"
    count = 1
    paths = collect_images(folder, count)
    
    print()
    print(f"Loaded {len(paths)} images")

    img = build_batch(paths)
    processed = process_image(img)
    preview_single(processed)
    
    # paths = collect_images(folder, 100)

    # img = build_batch(paths)

    # fingerprint = build_fingerprint(img)

    # show(fingerprint, len(paths))

if __name__ == "__main__":
    main()