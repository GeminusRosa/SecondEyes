import torch
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Custom transform to handle small images without upscaling
class SmartCropPad(object):
    """
    If image is smaller than 256x256: pad to 256x256
    If image is larger than 256x256: center crop to 256x256
    """
    def __init__(self, size=256, random_crop=False):
        self.size = size
        self.random_crop = random_crop
    
    def __call__(self, img):
        w, h = tuple(img.size)
        
        # If image is too small, pad it to 256x256
        if w < self.size or h < self.size:
            pad_w = max(0, self.size - w)
            pad_h = max(0, self.size - h)
            
            # Distribute padding evenly on both sides
            left = pad_w // 2
            top = pad_h // 2
            right = pad_w - left
            bottom = pad_h - top
            
            img = TF.pad(img, [left, top, right, bottom], fill=0)
        
        # Now crop to 256x256 if needed
        if w > self.size or h > self.size:
            if self.random_crop:
                img = T.RandomCrop(self.size)(img)
            else:
                img = T.CenterCrop(self.size)(img)
        
        return img

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
                    # Smart crop/pad: pads small images, crops large images
                    SmartCropPad(size=256, random_crop=True),

                    # Преобразуване в тензор.
                    T.ToTensor(),
                ]
            )

        else:
            # Трансформации при валидация.
            self.default_transform = T.Compose(
                [
                    # Smart crop/pad: pads small images, crops large images
                    SmartCropPad(size=256, random_crop=False),

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