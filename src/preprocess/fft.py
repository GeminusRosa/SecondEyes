import torch
import numpy as np
from scipy.ndimage import gaussian_filter

# Извикване на модула:
# fft = fft_magnitude(gaussian_residual(img)).float()

# Функция за извличане на остатък от изображение, чрез изваждане 
# на неговото Гаусово размазване от черно-бялото му представяне
def gaussian_residual(rgb):
    # Списък, в който ще се съхраняват остатъчните изображения
    output = []
    
    for img in rgb:
        # Преобразуване на изображението в черно-бяло чрез осредняване на каналите
        img_gray = img.detach().mean(0).cpu().numpy().astype(np.float32)
        
        # Прилагане на Гаусово размазване за премахване на нискочестотни компоненти
        img_blurred = gaussian_filter(img_gray, sigma=1)
        
        # Изчисляване на остатъчното изображение (високочестотни детайли)
        residual = img_gray - img_blurred
        
        # Добавяне на резултата към списъка
        output.append(residual)
    
    # Обединяване на всички остатъчни изображения в един тензор
    out = torch.from_numpy(np.stack(output).astype(np.float32)).to(rgb.device)
    
    # Добавяне на измерение (1) и връщане на резултата
    return out.unsqueeze(1).float()

# Функция за изчисляване на скалата на честотния спектър на изображение
def fft_magnitude(img):
    # Ако входът е NumPy масив, се преобразуване към PyTorch тензор
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    
    # Изчисляване на двумерното бързо преобразуване на Фурие
    f = torch.fft.fft2(img)
    
    # Преместване на нулевата честота в центъра на спектъра
    f = torch.fft.fftshift(f, dim=(-2, -1))
    
    # Изчисляване на логаритмичната скала на честотния спектър
    mag = torch.log1p(torch.abs(f))
    mag_min = mag.amin(dim=(-2, -1), keepdim=True)
    mag_max = mag.amax(dim=(-2, -1), keepdim=True)
    mag_norm = (mag - mag_min) / (mag_max - mag_min + 1e-12)
    
    # Връщане на резултата
    return mag_norm.float()