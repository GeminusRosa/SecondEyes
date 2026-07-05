import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        # Създаване на архитектура ResNet18 без предварително обучени тегла
        model = models.resnet18(weights=None)

        # Замяна на първия свиващ слой според броя на входните канали
        model.conv1 = nn.Conv2d(
            in_ch,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Премахване на крайния класификационен слой и запазване на енкодера
        self.net = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        # Извличане на признаци от входното изображение
        x = self.net(x)

        # Преобразуване на картата с признаци във вектор
        return torch.flatten(x, 1)