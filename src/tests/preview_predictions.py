import random
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.datasets.aria_dataset import ARIADataset
from src.models.fusion import ForensicNet

def show_validation_predictions(
    model_path="best_model.pth",
    val_dir="data/val",
    num_images=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = ARIADataset(val_dir, augment=False)

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
    )

    class_names = val_ds.classes

    model = ForensicNet(num_classes=len(class_names)).to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    indices = random.sample(
        range(len(val_ds)),
        min(num_images, len(val_ds)),
    )

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(18, 8),
    )

    axes = axes.flatten()

    with torch.no_grad():

        for ax, idx in zip(axes, indices):

            batch = val_ds[idx]

            image = batch["img"].unsqueeze(0).to(device)
            label = batch["label"].item()

            _, logits, _ = model(image)

            pred = torch.argmax(
                logits,
                dim=1,
            ).item()

            img = batch["img"].permute(1, 2, 0).cpu().numpy()

            ax.imshow(img)

            true_name = class_names[int(label)]
            pred_name = class_names[int(pred)]

            correct = label == pred

            ax.set_title(
                f"True: {true_name}\n" f"Pred: {pred_name}",
                color="green" if correct else "red",
                fontsize=16,
            )

            ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_validation_predictions()