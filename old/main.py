import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# ==========================================
# 1. ГЕОМЕТРИЧЕСКИЙ БАЗИС 2x2 (ВАША ЛОГИКА)
# ==========================================
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

# ==========================================
# 2. АРХИТЕКТУРА СЕТИ С ФИКСИРОВАННЫМ СЛОЕМ
# ==========================================
class CustomBasisNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Слой 1: Фиксированные 8 фильтров 2x2. Сканируем со stride=1
        self.fixed_conv = nn.Conv2d(1, 8, kernel_size=2, stride=2, padding=0, bias=False)
        fixed_weights = TEMPLATES.unsqueeze(1) # Подгоняем форму под (8, 1, 2, 2)
        self.fixed_conv.weight = nn.Parameter(fixed_weights, requires_grad=False)
        
        # Слой 2: Обучаемое расширение до 64 каналов
        self.expanded_conv = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # 27x27 -> 13x13
        )
        
        # Слой 3: Углубление абстракции до 128 каналов
        self.deep_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) # 13x13 -> 6x6
        )
        
        # Слой 4: Глобальная агрегация признаков и сужение до 10 классов
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.fixed_conv(x) 
        x = self.expanded_conv(x)
        x = self.deep_conv(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ==========================================
# 3. ФУНКЦИЯ ОБУЧЕНИЯ И СОХРАНЕНИЯ
# ==========================================
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство для обучения: {device}")
    
    # Загрузка данных без искажающей нормализации (чистый контраст)
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model = CustomBasisNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Обучаем 3 эпохи для надежной фиксации весов верхних слоев
    epochs = 10
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
        
    # Валидация
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nИтоговая точность на тесте: {accuracy:.2f}%")
    
    # СОХРАНЕНИЕ МОДЕЛИ
    # Мы сохраняем state_dict (только веса). Фиксированный первый слой запишется автоматически.
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Успех! Веса модели сохранены в файл: {MODEL_PATH}")

# ==========================================
# 4. ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ (ИНФЕРЕНС)
# ==========================================
def predict_image(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Файл весов {MODEL_PATH} не найден. Сначала запустите обучение!")
        return
        
    # Инициализируем архитектуру и загружаем сохраненные веса
    model = CustomBasisNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    # Подготовка стороннего изображения к формату сети (ЧБ, 28x28)
    try:
        img = Image.open(image_path).convert('L') # В градации серого
        img = img.resize((28, 28)) # Меняем размер под MNIST
        
        # Превращаем в тензор PyTorch и добавляем размерность батча (1, 1, 28, 28)
        transform = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            prediction = output.argmax(dim=1).item()
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()
            
        print(f"\n--- Анализ файла {image_path} ---")
        print(f"Результат предсказания: ЦИФРА {prediction}")
        print(f"Уверенность сети: {probabilities[prediction]*100:.2f}%")
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")

# ==========================================
# ТОЧКА ВХОДА ДЛЯ ЗАПУСКА
# ==========================================
if __name__ == "__main__":
    # Шаг 1: Запускаем обучение и сохраняем файл весов .pth
    train_pipeline()
    
    # Шаг 2: Демонстрация инференса. 
    # Если у вас есть своя картинка цифры, можете раскомментировать код ниже и передать путь к ней:
    # predict_image("my_handwritten_digit.png")
