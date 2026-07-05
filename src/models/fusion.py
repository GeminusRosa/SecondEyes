import torch.nn as nn

from .encoder import Encoder
from .attention import AttentionFusion
from src.preprocess.fft import gaussian_residual, fft_magnitude
from src.preprocess.dct import dct_transform
from src.preprocess.residual import residual_transform

class ForensicNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.img_encoder = Encoder(1)
        self.fft_encoder = Encoder(1)
        self.residual_encoder = Encoder(1)
        self.dct_encoder = Encoder(1)
        self.fusion = AttentionFusion()
        self.shared = nn.Sequential(
            nn.Linear(512 * 4, 512), 
            nn.ReLU(), 
            nn.Dropout(0.5)
        )

        self.binary_head = nn.Linear(512, 2)
        self.class_head = nn.Linear(512, num_classes)

    def forward(self, img):
        fft = fft_magnitude(gaussian_residual(img)).float()
        residual = residual_transform(img).float()
        dct = dct_transform(img).float()
        img_res = gaussian_residual(img)

        img_f = self.img_encoder(img_res)
        fft_f = self.fft_encoder(fft)
        residual_f = self.residual_encoder(residual)
        dct_f = self.dct_encoder(dct)
        fused, weights = self.fusion(img_f, fft_f, residual_f, dct_f)
        shared = self.shared(fused)

        return (
            self.binary_head(shared),
            self.class_head(shared),
            {
                "img": img_f,
                "fft": fft_f,
                "residual": residual_f,
                "dct": dct_f,
                "weights": weights,
            },
        )