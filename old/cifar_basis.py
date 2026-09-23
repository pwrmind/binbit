import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# СВЕРХБЫСТРОЕ ЗЕРКАЛО GOOGLE
# datasets.CIFAR10.url = "https://googleapis.com"

# 1. НАШИ СБАЛАНСИРОВАННЫЕ ФИЛЬТРЫ 2x2
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

MODEL_PATH = "cifar_basis_model.pth"

# Список классов CIFAR-10 для человекочитаемого вывода
CLASSES = ['самолет', 'автомобиль', 'птица', 'кошка', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']

# 2. АРХИТЕКТУРА ДЛЯ ЦВЕТНЫХ КАРТИНОК
class CifarBasisNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Слой 1: Фиксированный базис для RGB (groups=3 обрабатывает каналы изолированно)
        self.fixed_conv = nn.Conv2d(3, 24, kernel_size=2, stride=2, padding=0, bias=False, groups=3)
        fixed_weights = TEMPLATES.unsqueeze(1).repeat(3, 1, 1, 1) # Форма: (24, 1, 2, 2)
        self.fixed_conv.weight = nn.Parameter(fixed_weights, requires_grad=False)
        
        # Слой 2: Первое обучаемое расширение (24 -> 64 канала, размер 16x16 -> 8x8)
        self.layer1 = nn.Sequential(
            nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Слой 3: Углубление (64 -> 128 каналов, размер 8x8 -> 4x4)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Слой 4: Финальное абстрагирование
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Сжимаем до вектора в 256 чисел
        )
        
        self.classifier = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.fixed_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ==========================================
# 3. ФУНКЦИЯ ОБУЧЕНИЯ И СОХРАНЕНИЯ
# ==========================================
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Запуск эксперимента на CIFAR-10 ({device})...")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.CIFAR10(root='./data_cifar', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data_cifar', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model = CifarBasisNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Эпоха {epoch}/{epochs} | Средняя ошибка: {total_loss/len(train_loader):.4f}")
        
    # Проверка точности
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nТочность геометрического базиса на CIFAR-10: {accuracy:.2f}%")
    
    # СОХРАНЕНИЕ МОДЕЛИ
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Успех! Веса модели сохранены в файл: {MODEL_PATH}")

# ==========================================
# 4. ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ (ИНФЕРЕНС)
# ==========================================
def predict_cifar_image(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Файл весов {MODEL_PATH} не найден. Сначала запустите обучение!")
        return
        
    model = CifarBasisNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    try:
        # Открываем изображение, принудительно переводим в цветной RGB режим
        img = Image.open(image_path).convert('RGB')
        img = img.resize((32, 32), Image.Resampling.BILINEAR) # Сжимаем до размера CIFAR
        
        transform = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0) # Добавляем размерность батча (1, 3, 32, 32)
        
        with torch.no_grad():
            output = model(img_tensor)
            prediction = output.argmax(dim=1).item()
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()
            
        print(f"\n--- Анализ файла {image_path} ---")
        print(f"Результат предсказания: {CLASSES[prediction].upper()}")
        print(f"Уверенность сети: {probabilities[prediction]*100:.2f}%")
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")

if __name__ == "__main__":
    # Запуск основного цикла обучения и сохранения
    # train_pipeline()
    
    # Инференс: чтобы проверить свою картинку (например, фото машины или кота), 
    # положите её в папку, раскомментируйте строку ниже и укажите имя файла:
    predict_cifar_image("car.jpg")
