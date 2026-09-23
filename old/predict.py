import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.ndimage import maximum_filter # Фильтр для утолщения линий

# Ваши эталоны
TEMPLATES = torch.tensor([
    [[-1, -1], [-1, -1]],  # Инвертированный фон
    [[ 1,  1], [ 1,  1]],  # Плотность
    [[ 1,  1], [-1, -1]],  # Горизонтальная верхняя
    [[-1, -1], [ 1,  1]],  # Горизонтальная нижняя
    [[ 1, -1], [ 1, -1]],  # Вертикальная левая
    [[-1,  1], [-1,  1]],  # Вертикальная правая
    [[ 1, -1], [-1,  1]],  # Диагональ главная
    [[-1,  1], [ 1, -1]]   # Диагональ побочная
], dtype=torch.float32)

MODEL_PATH = "custom_basis_model.pth"

class CustomBasisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fixed_conv = nn.Conv2d(1, 8, kernel_size=2, stride=2, padding=0, bias=False)
        fixed_weights = TEMPLATES.unsqueeze(1)
        self.fixed_conv.weight = nn.Parameter(fixed_weights, requires_grad=False)
        
        self.expanded_conv = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.deep_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.fixed_conv(x) 
        x = self.expanded_conv(x)
        x = self.deep_conv(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def run_predict(image_path):
    if not os.path.exists(MODEL_PATH):
        print("Ошибка: файл весов не найден!")
        return
        
    model = CustomBasisNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    # 1. Открываем и сжимаем до 28x28
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28), Image.Resampling.BILINEAR)
    
    # 2. УТОЛЩАЕМ ЛИНИИ (Переводим в массив и делаем линии жирнее, как в MNIST)
    img_np = np.array(img)
    # Фильтр берет максимальное значение в окне 3х3, расширяя белый цвет на соседние пиксели
    img_thick = maximum_filter(img_np, size=3) 
    
    # (Опционально) Возвращаем в PIL, чтобы проверить глазами, если захотите сохранить
    img = Image.fromarray(img_thick)
    
    # 3. Переводим в тензор (чистый бинарный контраст 0-1)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)
    
    # Небольшой хак: делаем контраст жестче (все что выше 0.2 становится ярким)
    img_tensor = torch.where(img_tensor > 0.2, torch.tensor(1.0), torch.tensor(0.0))
    
    with torch.no_grad():
        output = model(img_tensor)
        prediction = output.argmax(dim=1).item()
        probabilities = torch.softmax(output, dim=1).squeeze().tolist()
        
    print(f"\nВы подали картинку: {image_path}")
    print(f"Робот утверждает, что это цифра: {prediction}")
    print(f"Вероятность (уверенность): {probabilities[prediction]*100:.2f}%")

if __name__ == "__main__":
    run_predict("my_digit.png") # Укажите имя вашей четверки
