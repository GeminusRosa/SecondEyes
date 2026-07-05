import pywt
import torch
import numpy as np

from .fft import fft_magnitude

# Извикване на модула:
# residual = residual_transform(img).float()

# Функция за потискане на шума чрез Wavelet-базирано Wiener филтриране
def wavelet_wiener(img, wavelet="db8", level=3):
    # Изчисляване на многомащабно вълново (wavelet) разлагане
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)

    # Запазване на апроксимационните коефициенти без промяна
    denoised_coeffs = [coeffs[0]]

    # Обработка на детайлните коефициенти за всяко ниво
    for details in coeffs[1:]:
        cH, cV, cD = details

        # Оценка на стандартното отклонение на шума
        sigma = np.median(np.abs(cD)) / 0.6745
        noise = sigma**2

        def wiener_subband(band):
            # Прилагане на Wiener филтриране върху отделна подлента
            var = np.maximum(band**2 - noise, 0)
            return (var / (var + noise + 1e-8)) * band

        # Филтриране на хоризонталните, вертикалните и диагоналните детайли
        denoised_coeffs.append((
            wiener_subband(cH),
            wiener_subband(cV),
            wiener_subband(cD)
        ))

    # Възстановяване на изображението от филтрираните коефициенти
    den = pywt.waverec2(denoised_coeffs, wavelet=wavelet)

    # Премахване на евентуално добавените гранични пиксели
    return den[: img.shape[0], : img.shape[1]]

# Функция за извличане на Wavelet-Wiener остатък от изображение
def residual_transform(rgb):
    # Списък за съхраняване на получените честотни представяния
    output = []

    for img in rgb:
        # Преобразуване на изображението в черно-бяло
        img_gray = img.detach().mean(0).cpu().numpy()

        # Потискане на шума чрез Wavelet-базирано Wiener филтриране
        den = wavelet_wiener(img_gray)

        # Изчисляване на остатъчното изображение
        residual = img_gray - den

        # Преобразуване на остатъка в честотната област чрез FFT
        output.append(fft_magnitude(residual))

    # Обединяване на резултатите в PyTorch тензор
    out = torch.from_numpy(np.stack(output).astype(np.float32)).to(rgb.device)

    # Добавяне на измерение (1) и връщане на резултата
    return out.unsqueeze(1).float()