import torch
import numpy as np
from scipy.fftpack import dct

# Извикване на модула:
# dct = dct_transform(img).float()

# Функция за изчисляване на 2D DCT
def dct2(img):
    return dct(dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")

# Функция за преобразуване на изображение чрез DCT
def dct_transform(rgb, gamma=0.10):
    # Списък за съхраняване на преобразуваните изображения
    output = []
    
    for img in rgb:
        # Преобразуване на изображението в черно-бяло чрез осредняване на каналите
        img_gray = img.detach().mean(0).cpu().numpy().astype(np.float32)
        
        # Изчисляване на абсолютните стойности на DCT коефициентите
        coeff = np.abs(dct2(img_gray))
        
        # Логаритмично мащабиране на коефициентите
        coeff = np.log(coeff + 1e-3)
        
        # Нормализация до нулева средна стойност и единично стандартно отклонение
        coeff -= np.mean(coeff)
        coeff /= np.std(coeff) + 1e-8
        
        # Прилагане на степенно преобразуване (gamma correction)
        # за подчертаване на по-слабите честотни компоненти
        coeff = np.sign(coeff) * np.abs(coeff) ** gamma
        
        # Добавяне на резултата към списъка
        output.append(coeff)
    
    # Обединяване на резултатите в PyTorch тензор
    out = torch.from_numpy(np.stack(output)).to(rgb.device)
    
    # Добавяне на измерение (1) и връщане на резултата
    return out.unsqueeze(1).float()