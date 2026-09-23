"""
best_basis_model.py
===================
Лучшая модель по результатам экспериментов:

    4 фиксированных геометрических фильтра 2x2
    + ReLU
    + stride=2 (неперекрывающиеся окна)

Поддерживает grayscale (1 канал) и RGB (3 канала, groups=3).
Работает на MNIST / FashionMNIST / KMNIST / CIFAR-10.
"""

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# ============================================================
# 1. ГЕОМЕТРИЧЕСКИЙ БАЗИС (4 независимых направления)
# ============================================================

TEMPLATES = torch.tensor([
    [[ 1,  1], [ 1,  1]],   # плотность / среднее
    [[ 1,  1], [-1, -1]],   # вертикальный перепад (верх-низ)
    [[ 1, -1], [ 1, -1]],   # горизонтальный перепад (лево-право)
    [[ 1, -1], [-1,  1]],   # диагональный перепад
], dtype=torch.float32)

# ============================================================
# 2. АРХИТЕКТУРА
# ============================================================

class BasisNet(nn.Module):
    """
    4 фиксированных фильтра 2x2 + ReLU + stride=2.
    Первый слой не обучается; всё остальное — обычные сверточные блоки.
    """
    def __init__(self, in_channels=1, num_classes=10, stride=2):
        super().__init__()
        groups = in_channels if in_channels > 1 else 1
        out_ch = 4 * groups  # 4 для grayscale, 12 для RGB

        # --- Фиксированный геометрический слой ---
        self.fixed_conv = nn.Conv2d(
            in_channels, out_ch,
            kernel_size=2, stride=stride, padding=0,
            bias=False, groups=groups
        )
        w = TEMPLATES.unsqueeze(1).repeat(groups, 1, 1, 1)  # (out_ch, 1, 2, 2)
        self.fixed_conv.weight = nn.Parameter(w, requires_grad=False)

        self.act = nn.ReLU()

        # --- Обучаемая часть ---
        self.features = nn.Sequential(
            nn.Conv2d(out_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fixed_conv(x)
        x = self.act(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================
# 3. ДАННЫЕ
# ============================================================

DATASETS = {
    "MNIST":        (datasets.MNIST,        "./data",       1, 10),
    "FashionMNIST": (datasets.FashionMNIST, "./data",       1, 10),
    "KMNIST":       (datasets.KMNIST,       "./data",       1, 10),
    "CIFAR10":      (datasets.CIFAR10,      "./data", 3, 10),
}


def get_loaders(name, batch_size=128):
    cls, root, in_ch, _ = DATASETS[name]
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST
    ])
    train_ds = cls(root=root, train=True,  download=True, transform=tf)
    test_ds  = cls(root=root, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1000, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader, in_ch


# ============================================================
# 4. ОБУЧЕНИЕ
# ============================================================

def train(dataset_name, epochs=5, lr=1e-3, save_path=None, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    train_loader, test_loader, in_ch = get_loaders(dataset_name)
    model = BasisNet(in_channels=in_ch).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Эпоха {ep}/{epochs} | loss = {total_loss / len(train_loader):.4f}")

    # --- Тест ---
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item()
    acc = 100 * correct / len(test_loader.dataset)
    print(f"  Точность на тесте ({dataset_name}): {acc:.2f}%")

    # --- Сохранение ---
    if save_path is None:
        save_path = f"basis_{dataset_name.lower()}.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "in_channels": in_ch,
        "dataset": dataset_name,
    }, save_path)
    print(f"  Модель сохранена: {save_path}")

    return model, acc


# ============================================================
# 5. ЗАГРУЗКА И ИНФЕРЕНС
# ============================================================

CLASSES = {
    "MNIST":        [str(i) for i in range(10)],
    "FashionMNIST": ['футболка', 'брюки', 'свитер', 'платье', 'пальто',
                     'сандалии', 'рубашка', 'кроссовки', 'сумка', 'ботинки'],
    "KMNIST":       [f'хирагана_{i}' for i in range(10)],
    "CIFAR10":      ['самолет', 'автомобиль', 'птица', 'кошка', 'олень',
                     'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик'],
}


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    in_ch = ckpt.get("in_channels", 1)
    dataset = ckpt.get("dataset", "MNIST")
    model = BasisNet(in_channels=in_ch)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, dataset


def predict(image_path, model_path):
    if not os.path.exists(model_path):
        print(f"Ошибка: {model_path} не найден. Сначала обучите модель.")
        return
    if not os.path.exists(image_path):
        print(f"Ошибка: файл {image_path} не найден.")
        return

    model, dataset = load_model(model_path)
    in_ch = 3 if dataset == "CIFAR10" else 1

    # --- Препроцессинг ---
    if in_ch == 1:
        img = Image.open(image_path).convert("L")
        size = 28
    else:
        img = Image.open(image_path).convert("RGB")
        size = 32
    img = img.resize((size, size), Image.BILINEAR)

    tf = transforms.ToTensor()
    x = tf(img).unsqueeze(0)

    # --- Инференс ---
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, 1).squeeze()
        pred = int(prob.argmax())

    labels = CLASSES.get(dataset, [str(i) for i in range(10)])
    print(f"\n--- {image_path} ---")
    print(f"Предсказание: {labels[pred]}")
    print(f"Уверенность:  {prob[pred]*100:.2f}%")
    print("\nТоп-3:")
    top3 = torch.topk(prob, 3)
    for p, i in zip(top3.values.tolist(), top3.indices.tolist()):
        print(f"  {labels[i]:<12} {p*100:6.2f}%")


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

if __name__ == "__main__":
    # --- Обучить на MNIST (быстро, 5 эпох) ---
    train("MNIST", epochs=5)

    # --- Или на FashionMNIST / KMNIST / CIFAR10 ---
    # train("FashionMNIST", epochs=5)
    # train("KMNIST",        epochs=5)
    # train("CIFAR10",       epochs=15)

    # --- Инференс по картинке ---
    # predict("my_digit.png", "basis_mnist.pth")