import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()

        # Мрежа за изчисляване на теглата на вниманието
        self.att = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, img, fft, residual, dct):
        # Обединяване на признаците от четирите входни клона
        x = torch.cat([img, fft, residual, dct], dim=1)

        # Изчисляване на теглата за всеки клон
        w = self.att(x)

        # Претегляне на признаците и обединяването им в един вектор
        fused = torch.cat(
            [
                img * w[:, 0:1],
                fft * w[:, 1:2],
                residual * w[:, 2:3],
                dct * w[:, 3:4]
            ],
            dim=1
        )

        # Връщане на обединените признаци и изчислените тегла
        return fused, w