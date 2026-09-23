"""
final_experiments.py
====================
Закрываем открытые вопросы:

  Для MNIST и FashionMNIST:
     A. 15 эпох, без аугментации, фикс. базис  (референс)
     B. 25 эпох, без аугментации, фикс. базис
     C. 25 эпох, +аугментация,    фикс. базис
     D. 25 эпох, +аугментация,    обучаемый слой, инициализированный шаблонами

  Для CIFAR-10:
     E. 25 эпох, +аугментация, фикс. базис (groups=3, 12 каналов)
     F. 25 эпох, +аугментация, обучаемый слой (init шаблонами)

Без нормализации — мы доказали, что она вредит базису.
С CosineAnnealingLR везде.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time


# ============================================================
# 1. ФИКСИРОВАННЫЙ БАЗИС
# ============================================================

TEMPLATES = torch.tensor([
    [[ 1,  1], [ 1,  1]],
    [[ 1,  1], [-1, -1]],
    [[ 1, -1], [ 1, -1]],
    [[ 1, -1], [-1,  1]],
], dtype=torch.float32)


# ============================================================
# 2. МОДЕЛЬ
# ============================================================

class BasisNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, stride=2, trainable=False):
        super().__init__()
        groups = in_channels if in_channels > 1 else 1
        self.out_ch = 4 * groups

        self.fixed_conv = nn.Conv2d(in_channels, self.out_ch, 2, stride=stride,
                                    padding=0, bias=False, groups=groups)
        w = TEMPLATES.unsqueeze(1).repeat(groups, 1, 1, 1)
        self.fixed_conv.weight = nn.Parameter(w, requires_grad=trainable)

        self.act = nn.ReLU()
        self.features = nn.Sequential(
            nn.Conv2d(self.out_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.act(self.fixed_conv(x))
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


# ============================================================
# 3. ДАННЫЕ + АУГМЕНТАЦИЯ
# ============================================================

def get_transforms(dataset_name, augment):
    if dataset_name == "CIFAR10":
        base = [transforms.ToTensor()]
        if augment:
            base = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        return transforms.Compose(base)

    # MNIST / FashionMNIST
    if augment:
        return transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
        ])
    return transforms.Compose([transforms.ToTensor()])


def get_loaders(dataset_name, augment, batch_size=128):
    tf_train = get_transforms(dataset_name, augment=augment)
    tf_test  = transforms.Compose([transforms.ToTensor()])

    if dataset_name == "MNIST":
        cls, root, in_ch = datasets.MNIST, "./data", 1
    elif dataset_name == "FashionMNIST":
        cls, root, in_ch = datasets.FashionMNIST, "./data", 1
    elif dataset_name == "CIFAR10":
        cls, root, in_ch = datasets.CIFAR10, "./data_cifar", 3
    else:
        raise ValueError(dataset_name)

    train_ds = cls(root=root, train=True,  download=True, transform=tf_train)
    test_ds  = cls(root=root, train=False, download=True, transform=tf_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1000, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader, in_ch


# ============================================================
# 4. ОБУЧЕНИЕ
# ============================================================

def run_experiment(dataset_name, epochs, augment, trainable, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, in_ch = get_loaders(dataset_name, augment=augment)
    model = BasisNet(in_channels=in_ch, trainable=trainable).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable_params, lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()

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
        sched.step()
        # Печатаем каждые 5 эпох, чтобы лог не разрастался
        if ep % 5 == 0 or ep == epochs:
            print(f"    эпоха {ep:>2}/{epochs} | loss = {total_loss/len(train_loader):.4f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.time() - t0

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item()

    acc = 100 * correct / len(test_loader.dataset)
    return acc, train_time


# ============================================================
# 5. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------- MNIST / FashionMNIST ----------
    plans_mnist = [
        ("A. 15 эп, без аугм, фикс",     15, False, False),
        ("B. 25 эп, без аугм, фикс",     25, False, False),
        ("C. 25 эп, +аугм,    фикс",     25, True,  False),
        ("D. 25 эп, +аугм,    обуч. init", 25, True, True),
    ]

    summary = {}

    for ds in ["MNIST", "FashionMNIST"]:
        print(f"\n{'='*72}\nНАБОР: {ds}\n{'='*72}")
        rows = []
        for label, ep, aug, tr in plans_mnist:
            print(f"\n>>> {label}")
            acc, t = run_experiment(ds, epochs=ep, augment=aug, trainable=tr)
            print(f"    Точность: {acc:.2f}%  |  Время: {t:.1f}с")
            rows.append((label, acc, t))
        summary[ds] = rows

    # ---------- CIFAR-10 ----------
    plans_cifar = [
        ("E. 25 эп, +аугм, фикс",       25, True, False),
        ("F. 25 эп, +аугм, обуч. init", 25, True, True),
    ]

    print(f"\n{'='*72}\nНАБОР: CIFAR-10\n{'='*72}")
    rows = []
    for label, ep, aug, tr in plans_cifar:
        print(f"\n>>> {label}")
        acc, t = run_experiment("CIFAR10", epochs=ep, augment=aug, trainable=tr)
        print(f"    Точность: {acc:.2f}%  |  Время: {t:.1f}с")
        rows.append((label, acc, t))
    summary["CIFAR10"] = rows

    # ---------- Сводка ----------
    print("\n\n" + "=" * 82)
    print("СВОДКА".center(82))
    print("=" * 82)
    for ds in ["MNIST", "FashionMNIST", "CIFAR10"]:
        print(f"\n### {ds} ###")
        print(f"{'Конфигурация':<32}{'Точность':>12}{'Время':>10}")
        print("-" * 54)
        for label, acc, t in summary[ds]:
            print(f"{label:<32}{acc:>11.2f}%{t:>9.1f}с")