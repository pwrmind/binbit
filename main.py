import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. ОПРЕДЕЛЯЕМ ВАШИ 6 ЭТАЛОНОВ 2x2
templates = torch.tensor([
    [[0, 0], [0, 0]],  # Горизонтальная верхняя
    [[1, 1], [1, 1]],  # Горизонтальная верхняя
    [[1, 1], [0, 0]],  # Горизонтальная верхняя
    [[0, 0], [1, 1]],  # Горизонтальная нижняя
    [[1, 0], [1, 0]],  # Вертикальная левая
    [[0, 1], [0, 1]],  # Вертикальная правая
    [[1, 0], [0, 1]],  # Диагональ главная
    [[0, 1], [1, 0]]   # Диагональ побочная
], dtype=torch.float32)

# 2. АРХИТЕКТУРА СЕТИ
class CustomBasisNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Слой 1: Зафиксированные 6 фильтров (без обучения)
        self.fixed_conv = nn.Conv2d(1, 8, kernel_size=2, stride=2, padding=0, bias=False)
        fixed_weights = templates.unsqueeze(1) # Изменяем форму под (6, 1, 2, 2)
        self.fixed_conv.weight = nn.Parameter(fixed_weights, requires_grad=False)
        
        # Слой 2: Расширение (обучаемый слой, комбинирует примитивы)
        self.expanded_conv = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # Сжимаем 14x14 -> 7x7
        )
        
        # Слой 3: Еще один шаг абстракции
        self.deep_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Сужаем до вектора (Global Average Pooling)
        )
        
        # Слой 4: Финальное сужение до 10 классов
        self.classifier = nn.Linear(64, 10)
        
    def forward(self, x):
        # 1. Извлекаем ваши геометрические примитивы
        x = self.fixed_conv(x) 
        # 2. Расширяем и комбинируем их
        x = self.expanded_conv(x)
        # 3. Углубляем абстракцию
        x = self.deep_conv(x)
        # 4. Плоский вектор на классификатор
        x = torch.flatten(x, 1)
        return self.classifier(x)

# 3. ПОДГОТОВКА ДАННЫХ (MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Нормализация для MNIST
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 4. НАСТРОЙКА ОБУЧЕНИЯ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomBasisNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. ЦИКЛ ОБУЧЕНИЯ (2 эпохи для демонстрации)
print(f"Запуск обучения на устройстве: {device}\n")
for epoch in range(1, 3):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Эпоха {epoch} завершена. Средняя ошибка: {total_loss/len(train_loader):.4f}")

# 6. ПРОВЕРКА ТОЧНОСТИ
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

accuracy = 100. * correct / len(test_loader.dataset)
print(f"\nТочность на тестовой выборке: {accuracy:.2f}%")
