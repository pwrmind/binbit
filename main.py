import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 1. ВАШИ 8 СБАЛАНСИРОВАННЫХ ФИЛЬТРОВ 2x2
templates = torch.tensor([
    [[-1, -1], [-1, -1]],  # Инвертированный фон
    [[ 1,  1], [ 1,  1]],  # Плотность
    [[ 1,  1], [-1, -1]],  # Горизонтальная верхняя
    [[-1, -1], [ 1,  1]],  # Горизонтальная нижняя
    [[ 1, -1], [ 1, -1]],  # Вертикальная левая
    [[-1,  1], [-1,  1]],  # Вертикальная правая
    [[ 1, -1], [-1,  1]],  # Диагональ главная
    [[-1,  1], [ 1, -1]]   # Диагональ побочная
], dtype=torch.float32)

# 2. ФИКСИРОВАННЫЙ ЭКСТРАКТОР (БЕЗ ОБУЧЕНИЯ)
class PureBasisExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fixed_conv = nn.Conv2d(1, 8, kernel_size=2, stride=1, padding=0, bias=False)
        fixed_weights = templates.unsqueeze(1)
        self.fixed_conv.weight = nn.Parameter(fixed_weights, requires_grad=False)
        
    def forward(self, x):
        # На выходе получаем карту 8x27x27
        x = self.fixed_conv(x)
        # Выпрямляем в вектор размера 5832
        return torch.flatten(x, 1)

# 3. ПОДГОТОВКА ДАННЫХ (Чистый бинарный контраст, без сдвигов нормализации)
transform = transforms.Compose([transforms.ToTensor()])

# Ограничим выборку для k-NN, так как он считает расстояния "в лоб" и на CPU 
# обработка всех 60 000 картинок займет много времени. Возьмем подвыборку.
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Возьмем 10 000 картинок для базы знаний и 2 000 для теста (этого более чем достаточно)
train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2000, shuffle=False)

# Извлекаем тензоры
X_train_raw, y_train = next(iter(train_loader))
X_test_raw, y_test = next(iter(test_loader))

# 4.ПРОПУСКАЕМ ЧЕРЕЗ ВАШ БАЗИС
extractor = PureBasisExtractor()
extractor.eval() # Переводим в режим оценки

print("Извлечение структурных признаков через ваши фильтры...")
with torch.no_grad():
    X_train_features = extractor(X_train_raw).numpy()
    X_test_features = extractor(X_test_raw).numpy()

print(f"Размер полученных векторов: {X_train_features.shape[1]} признаков на картинку.")

# 5. КЛАССИФИКАЦИЯ БЕЗ ГРАДИЕНТОВ (k-NN)
print("Запуск k-NN классификатора (поиск по геометрии)...")
# Ищем 3-х самых близких соседей по Евклидовому расстоянию
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1) 
knn.fit(X_train_features, y_train.numpy())

# Предсказание
y_pred = knn.predict(X_test_features)

# Результат
accuracy = accuracy_score(y_test.numpy(), y_pred)
print(f"\nТочность БЕЗ ОБУЧЕНИЯ СЕТИ (чистая логика базиса + k-NN): {accuracy * 100:.2f}%")
