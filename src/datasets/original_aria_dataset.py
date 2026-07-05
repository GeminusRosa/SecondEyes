import torch
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as T

# Клас за зареждане и предварителна обработка на изображенията
# от набора от данни ARIA.
class ARIADataset(Dataset):

    # Инициализира набора от данни, като зарежда структурата
    # на директориите и подготвя необходимите трансформации.
    def __init__(self, root_dir, transform=None, augment=False):
        # Запазване на пътя до основната директория.
        self.root = Path(root_dir)

        # Запазване на подадените трансформации.
        self.transform = transform

        # Определяне дали ще се използва уголемяване
        # на обучаващите данни (Data Augmentation).
        self.augment = augment

        # Списък, съдържащ всички изображения и съответните им етикети.
        self.samples = []

        # Извличане имената на всички поддиректории,
        # които представляват отделните класове.
        self.classes = sorted(
            [p.name for p in self.root.iterdir() if p.is_dir()]
        )

        # Създаване на съответствие между името
        # на всеки клас и неговия числов идентификатор.
        self.class_to_idx = {
            c: i for i, c in enumerate(self.classes)
        }

        # Обхождане на всички класове.
        for cls in self.classes:
            folder = self.root / cls

            # Обхождане на всички изображения в съответната директория.
            for img in folder.iterdir():
                # Добавяне само на изображения с поддържан файлов формат.
                if img.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                ]:

                    # Запазване на пътя до изображението
                    # и числовия етикет на неговия клас.
                    self.samples.append(
                        (str(img), self.class_to_idx[cls])
                    )

        # Извеждане на броя успешно заредени изображения.
        print(f"Loaded {len(self.samples)} images")

        # Дефиниране на трансформации за увеличаване
        # на разнообразието на обучаващите данни.
        self.augment_transform = T.Compose(
            [
                # Случайно хоризонтално обръщане.
                T.RandomHorizontalFlip(),

                # Произволна промяна на яркостта, контраста и наситеността.
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                ),
            ]
        )

        # Избор на трансформации според режима на работа (обучение или валидация).
        if augment:
            # Трансформации при обучение.
            self.default_transform = T.Compose(
                [
                    # Преоразмеряване до 512×512 пиксела.
                    T.Resize((512, 512)),

                    # Случайно изрязване на област 256×256 пиксела.
                    T.RandomCrop(256),

                    # Преобразуване в тензор.
                    T.ToTensor(),
                ]
            )

        else:
            # Трансформации при валидация.
            self.default_transform = T.Compose(
                [
                    # Преоразмеряване до 512×512 пиксела.
                    T.Resize((512, 512)),

                    # Централно изрязване до размер 256×256 пиксела.
                    T.CenterCrop(256),

                    # Преобразуване в тензор.
                    T.ToTensor(),
                ]
            )

    # Връща общия брой изображения в набора от данни.
    def __len__(self):
        return len(self.samples)

    # Зарежда изображение по неговия индекс, прилага необходимите трансформации
    # и подготвя данните за невронната мрежа.
    def __getitem__(self, idx):
        # Извличане на пътя до изображението и неговия етикет.
        path, label = self.samples[idx]

        # Зареждане на изображението и преобразуването му в RGB формат.
        img = Image.open(path).convert("RGB")

        # Прилагане на Data Augmentation, ако е активирано.
        if self.augment:
            img = self.augment_transform(img)

        # Прилагане на потребителски дефинирани трансформации.
        if self.transform:
            img = self.transform(img)

        # Ако няма подадени трансформации, използват се стандартните.
        else:
            img = self.default_transform(img)

        # Връщане на обработеното изображение, числовия етикет и пътя до файла.
        return {
            "img": img,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(path),
        }