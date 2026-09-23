"""
compare_improvements.py
=======================
Проверяем вклад улучшений поверх лучшей модели (4 фикс + ReLU + stride=2):

    1. Базовая (как было)                       — baseline
    2. + Нормализация данных                    — normalize
    3. + Нормализация + CosineAnnealingLR       — sched
    4. + Нормализация + Cosine + больше эпох    — long

Все конфигурации с одинаковым seed — сравнение честное.
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
    def __init__(self, in_channels=1, num_classes=10, stride=2):
        super().__init__()
        groups = in_channels if in_channels > 1 else 1
        out_ch = 4 * groups

        self.fixed_conv = nn.Conv2d(in_channels, out_ch, 2, stride=stride,
                                    padding=0, bias=False, groups=groups)
        w = TEMPLATES.unsqueeze(1).repeat(groups, 1, 1, 1)
        self.fixed_conv.weight = nn.Parameter(w, requires_grad=False)
        self.act = nn.ReLU()

        self.features = nn.Sequential(
            nn.Conv2d(out_ch, 64, 3, padding=1),
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
# 3. ДАННЫЕ
# ============================================================

# Средние и std для нормализации (стандартные для MNIST-семейства)
NORM_STATS = {
    "MNIST":        ((0.1307,), (0.3081,)),
    "FashionMNIST": ((0.2860,), (0.3530,)),
    "KMNIST":       ((0.1918,), (0.3483,)),
}


def get_loaders(name, normalize=False, batch_size=128):
    if name == "MNIST":
        cls, root = datasets.MNIST, "./data"
    elif name == "FashionMNIST":
        cls, root = datasets.FashionMNIST, "./data"
    elif name == "KMNIST":
        cls, root = datasets.KMNIST, "./data"
    else:
        raise ValueError(name)

    tf_list = [transforms.ToTensor()]
    if normalize:
        mean, std = NORM_STATS[name]
        tf_list.append(transforms.Normalize(mean, std))
    tf = transforms.Compose(tf_list)

    train_ds = cls(root=root, train=True,  download=True, transform=tf)
    test_ds  = cls(root=root, train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1000, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ============================================================
# 4. ОБУЧЕНИЕ
# ============================================================

def run_experiment(dataset_name, epochs, normalize, use_scheduler, seed=42, lr=1e-3):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loaders(dataset_name, normalize=normalize)
    model = BasisNet(in_channels=1).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=lr)
    crit = nn.CrossEntropyLoss()

    sched = None
    if use_scheduler:
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

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
        if sched is not None:
            sched.step()

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

    # (имя, эпох, normalize, scheduler)
    configs = [
        ("1. Базовая (5 эпох)",              5,  False, False),
        ("2. + Нормализация (5 эпох)",       5,  True,  False),
        ("3. + Норм + Cosine (5 эпох)",      5,  True,  True),
        ("4. + Норм + Cosine (15 эпох)",     15, True,  True),
    ]

    datasets_to_run = ["MNIST", "FashionMNIST"]

    summary = {}
    for ds in datasets_to_run:
        print(f"\n{'='*70}\nНАБОР: {ds}\n{'='*70}")
        rows = []
        for name, ep, norm, sched in configs:
            print(f"\n>>> {name}")
            acc, t = run_experiment(ds, epochs=ep, normalize=norm, use_scheduler=sched)
            print(f"    Точность: {acc:.2f}%  |  Время: {t:.1f}с")
            rows.append((name, acc, t))
        summary[ds] = rows

    # --- Итог ---
    print("\n\n" + "=" * 78)
    print("СВОДКА".center(78))
    print("=" * 78)
    for ds in datasets_to_run:
        print(f"\n### {ds} ###")
        print(f"{'Конфигурация':<32}{'Точность':>12}{'Время':>10}")
        print("-" * 54)
        for name, acc, t in summary[ds]:
            print(f"{name:<32}{acc:>11.2f}%{t:>9.1f}с")