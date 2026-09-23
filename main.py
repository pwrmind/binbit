import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# ============================================================
# НАБОРЫ ФИЛЬТРОВ
# ============================================================

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

TEMPLATES_4 = torch.tensor([
    [[ 1,  1], [ 1,  1]],
    [[ 1,  1], [-1, -1]],
    [[ 1, -1], [ 1, -1]],
    [[ 1, -1], [-1,  1]],
], dtype=torch.float32)

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

def get_datasets(name):
    transform = transforms.Compose([transforms.ToTensor()])
    if name == "MNIST":
        tr = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
        te = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif name == "FashionMNIST":
        tr = datasets.FashionMNIST(root='./data', train=True,  download=True, transform=transform)
        te = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif name == "KMNIST":
        tr = datasets.KMNIST(root='./data', train=True,  download=True, transform=transform)
        te = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(name)
    return tr, te


def train_and_eval(templates, stride, use_relu, dataset_name, epochs=3, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = get_datasets(dataset_name)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)

    model = Net(templates, stride=stride, use_relu=use_relu).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=0.001)
    crit = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.time() - t0

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()

    acc = 100 * correct / len(test_ds)
    n_params = sum(p.numel() for p in trainable)
    return acc, train_time, n_params


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = [
        ("8 фильтров,     stride=2, без ReLU", TEMPLATES_8,     2, False),
        ("4 фильтра,      stride=2, без ReLU", TEMPLATES_4,     2, False),
        ("4×2 контроль,   stride=2, без ReLU", TEMPLATES_4_DUP, 2, False),
        ("8 фильтров,     stride=2, с ReLU",   TEMPLATES_8,     2, True),
        ("4 фильтра,      stride=2, с ReLU",   TEMPLATES_4,     2, True),
        ("8 фильтров,     stride=1, без ReLU", TEMPLATES_8,     1, False),
        ("4 фильтра,      stride=1, без ReLU", TEMPLATES_4,     1, False),
    ]

    datasets_to_run = ["MNIST", "FashionMNIST", "KMNIST"]
    epochs = 5   # CUDA быстрая, можно позволить больше

    all_results = {}
    for ds_name in datasets_to_run:
        print(f"\n{'='*70}\nНАБОР: {ds_name}  (train=60k, test=t10k)\n{'='*70}")
        results = []
        for name, tmpl, stride, relu in configs:
            acc, t, n_params = train_and_eval(tmpl, stride, relu, ds_name, epochs=epochs)
            print(f"  {name:<38} точн={acc:6.2f}%  время={t:5.1f}с  параметров={n_params:>7,}")
            results.append((name, acc, t, n_params))
        all_results[ds_name] = results

    # Итоговые таблицы
    for ds_name in datasets_to_run:
        print(f"\n\n### {ds_name} (t10k) ###")
        print(f"{'Конфигурация':<40}{'Точность':>10}{'Время':>9}{'Параметры':>12}")
        print("-" * 72)
        for name, acc, t, n_params in all_results[ds_name]:
            print(f"{name:<40}{acc:>9.2f}%{t:>8.1f}с{n_params:>12,}")