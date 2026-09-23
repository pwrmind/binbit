import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# ============================================================
# НАБОРЫ ФИЛЬТРОВ
# ============================================================

# Оригинальные 8 шаблонов (4 независимых + их отрицания)
TEMPLATES_8 = torch.tensor([
    [[-1, -1], [-1, -1]],
    [[ 1,  1], [ 1,  1]],
    [[ 1,  1], [-1, -1]],
    [[-1, -1], [ 1,  1]],
    [[ 1, -1], [ 1, -1]],
    [[-1,  1], [-1,  1]],
    [[ 1, -1], [-1,  1]],
    [[-1,  1], [ 1, -1]]
], dtype=torch.float32)

# 4 независимых (базис пространства 2x2)
TEMPLATES_4 = torch.tensor([
    [[ 1,  1], [ 1,  1]],   # плотность
    [[ 1,  1], [-1, -1]],   # вертикальный перепад
    [[ 1, -1], [ 1, -1]],   # горизонтальный перепад
    [[ 1, -1], [-1,  1]],   # диагональный перепад
], dtype=torch.float32)

# Контроль: 4 фильтра продублированы (rank=4, но 8 каналов)
TEMPLATES_4_DUP = torch.cat([TEMPLATES_4, TEMPLATES_4], dim=0)


# ============================================================
# АРХИТЕКТУРА
# ============================================================

class Net(nn.Module):
    def __init__(self, templates, stride=2, use_relu=False):
        super().__init__()
        n = templates.shape[0]
        self.fixed_conv = nn.Conv2d(1, n, kernel_size=2, stride=stride, padding=0, bias=False)
        self.fixed_conv.weight = nn.Parameter(templates.unsqueeze(1), requires_grad=False)
        self.use_relu = use_relu

        self.body = nn.Sequential(
            nn.Conv2d(n, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fixed_conv(x)
        if self.use_relu:
            x = torch.relu(x)
        x = self.body(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================
# ОБУЧЕНИЕ + ОЦЕНКА
# ============================================================

def train_and_eval(templates, stride, use_relu, epochs=3, seed=42):
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

    model = Net(templates, stride=stride, use_relu=use_relu).to(device)

    # обучаем только нефиксированные параметры
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=0.001)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
    train_time = time.time() - t0

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()

    acc = 100 * correct / len(test_ds)
    n_params = sum(p.numel() for p in trainable)
    return acc, train_time, n_params


# ============================================================
# ЗАПУСК ВСЕХ КОНФИГУРАЦИЙ
# ============================================================

if __name__ == "__main__":
    configs = [
        ("8 фильтров (оригинал), stride=2, без ReLU", TEMPLATES_8,     2, False),
        ("4 фильтра (базис),     stride=2, без ReLU", TEMPLATES_4,     2, False),
        ("4 фильтра × 2 (контроль), stride=2, без ReLU", TEMPLATES_4_DUP, 2, False),
        ("8 фильтров (оригинал), stride=2, с ReLU",   TEMPLATES_8,     2, True),
        ("4 фильтра (базис),     stride=2, с ReLU",   TEMPLATES_4,     2, True),
        ("8 фильтров (оригинал), stride=1, без ReLU", TEMPLATES_8,     1, False),
        ("4 фильтра (базис),     stride=1, без ReLU", TEMPLATES_4,     1, False),
    ]

    results = []
    for name, tmpl, stride, relu in configs:
        print(f"\n>>> {name}")
        acc, t, n_params = train_and_eval(tmpl, stride, relu, epochs=3)
        print(f"    Точность: {acc:.2f}%  |  Время: {t:.1f}с  |  Обучаемых параметров: {n_params:,}")
        results.append((name, acc, t, n_params))

    print("\n" + "=" * 90)
    print(f"{'Конфигурация':<48} {'Точн.':>7} {'Время':>7} {'Парам.':>12}")
    print("=" * 90)
    for name, acc, t, n_params in results:
        print(f"{name:<48} {acc:>6.2f}% {t:>6.1f}с {n_params:>12,}")