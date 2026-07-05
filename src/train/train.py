import os
import torch
import torch.nn as nn
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.datasets.aria_dataset import ARIADataset
from src.models.fusion import ForensicNet

# Изчисляване на точността на класификацията за текущата минипартида.
def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

# Анализ на средните Attention тегла за всеки клас изображения.
def analyze_attention_weights(model, val_loader, class_names, device):
    # Превключване на модела в режим на оценяване.
    model.eval()

    weights_by_class = defaultdict(list)

    # Изключване на изчисляването на градиенти.
    with torch.no_grad():
        # Обхождане на валидационния набор.
        for batch in val_loader:
            img = batch["img"].to(device)
            labels = batch["label"]

            # Получаване на Attention теглата.
            _, _, feats = model(img)

            weights = feats["weights"].cpu()

            # Групиране на теглата според класа на изображението.
            for i in range(len(labels)):
                cls_name = class_names[labels[i].item()]

                weights_by_class[cls_name].append(weights[i])

    rows = []

    # Изчисляване на средните Attention тегла за всеки клас.
    for cls_name, vals in weights_by_class.items():
        mean_w = torch.stack(vals).mean(0)

        rows.append(
            {
                "class": cls_name,
                "image": mean_w[0].item(),
                "fft": mean_w[1].item(),
                "residual": mean_w[2].item(),
                "dct": mean_w[3].item(),
            }
        )

    df = pd.DataFrame(rows)
    print(df)

    # Записване на анализа във CSV файл.
    df.to_csv("attention_analysis.csv", index=False)

# Основна функция, реализираща обучението на модела.
def train():
    # Автоматичен избор между CPU и GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Зареждане на обучаващия и валидационния набор.
    train_ds = ARIADataset("data/train", augment=True)

    val_ds = ARIADataset("data/val", augment=False)

    # Създаване на DataLoader обектите.
    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Определяне броя на класовете.
    num_classes = len(train_ds.classes)

    # Запазване индекса на реалните изображения.
    real_idx = train_ds.class_to_idx["Real"]

    # Създаване на архитектурата.
    model = ForensicNet(num_classes).to(device)

    # Дефиниране на функцията на загубите.
    criterion = nn.CrossEntropyLoss()

    # Използване на оптимизатора AdamW.
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=5e-4)

    # Автоматично намаляване на learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_acc = 0
    start_epoch = 0

    # Проверка дали съществува предварително обучен модел.
    if os.path.exists("best_model.pth"):
        checkpoint = torch.load("best_model.pth", map_location=device)

        # Зареждане състоянието на модела.
        model.load_state_dict(checkpoint["model_state"])

        optimizer.load_state_dict(checkpoint["optimizer_state"])

        scheduler.load_state_dict(checkpoint["scheduler_state"])

        start_epoch = checkpoint["epoch"] + 1

        best_val_acc = checkpoint["val_acc"]

        print(f"Resuming from epoch " f"{start_epoch}")

    else:
        print("No checkpoint found. " "Starting fresh.")

    # Параметри на Early Stopping.
    patience = 10
    wait = 0
    extra_epochs = 25

    # Основен цикъл на обучението.
    for epoch in range(start_epoch, start_epoch + extra_epochs):

        print(f"\nEpoch {epoch + 1}")

        model.train()

        train_loss = 0
        train_acc = 0

        train_loop = tqdm(train_loader, desc="Train", bar_format="{desc}: {percentage:3.0f}%")

        # Обучение върху всички минипартиди.
        for batch in train_loop:
            img = batch["img"].to(device)
            labels = batch["label"].to(device)

            # Формиране на бинарните етикети.
            binary = (labels != real_idx).long()

            optimizer.zero_grad()

            # Forward propagation.
            bin_out, cls_out, _ = model(img)

            # Изчисляване на общата функция на загубите.
            loss = criterion(bin_out, binary) + criterion(cls_out, labels)

            # Backpropagation.
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(cls_out, labels)

            train_loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Преминаване към валидация.
        model.eval()

        val_acc = 0
        val_loss = 0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc="Validation", bar_format="{desc}: {percentage:3.0f}%")

            for batch in val_loop:
                img = batch["img"].to(device)
                labels = batch["label"].to(device)

                binary = (labels != real_idx).long()

                bin_out, cls_out, _ = model(img)

                loss = criterion(bin_out, binary) + criterion(cls_out, labels)

                val_loss += loss.item()
                val_acc += accuracy(cls_out, labels)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print()
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc : {train_acc:.4f}")
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Val Acc   : {val_acc:.4f}")

        # Актуализиране на learning rate.
        scheduler.step(val_acc)

        # Запазване на най-добрия модел.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "classes": train_ds.classes,
                },
                "best_model.pth",
            )

            print("Saved best model!")

        else:
            wait += 1

            # Прекратяване при липса на подобрение.
            if wait >= patience:
                print("\nEarly stopping.")
                break

    # Зареждане на най-добрата версия на модела.
    checkpoint = torch.load("best_model.pth", map_location=device)

    model.load_state_dict(checkpoint["model_state"])

    # Анализ на Attention механизма.
    analyze_attention_weights(model, val_loader, train_ds.classes, device)

# Стартиране на обучението.
if __name__ == "__main__":
    train()