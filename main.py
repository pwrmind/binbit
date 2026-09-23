"""
analyst_check.py
================
Проверяем гипотезы аналитика против нашей проверенной конфигурации.

Конфигурации:
    1. Baseline              : 4 фикс + ReLU                  (референс)
    2. +BN после fixed_conv  : Conv -> BN -> ReLU
    3. +abs вместо ReLU      : Conv -> |x|
    4. +норм. шаблоны /2     : 4 фикс/2 + ReLU
    5. +LeakyReLU            : Conv -> LeakyReLU (0.1)
    6. Всё сразу             : Conv/2 -> BN -> abs

Всё без нормализации данных, 15 эпох, CosineAnnealingLR.
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
# ШАБЛОНЫ (нормальный и нормализованный)
# ============================================================

TEMPLATES = torch.tensor([
    [[ 1,  1], [ 1,  1]],
    [[ 1,  1], [-1, -1]],
    [[ 1, -1], [ 1, -1]],
    [[ 1, -1], [-1,  1]],
], dtype=torch.float32)

TEMPLATES_NORM = TEMPLATES / 2.0


# ============================================================
# ГИБКАЯ МОДЕЛЬ
# ============================================================

class BasisNet(nn.Module):
    """
    cfg:
      templates   : тензор (4, 2, 2)
      use_bn      : BatchNorm после fixed_conv
      activation  : 'relu' | 'abs' | 'leaky'
    """
    def __init__(self, in_channels=1, num_classes=10, stride=2,
                 templates=TEMPLATES, use_bn=False, activation='relu'):
        super().__init__()
        groups = in_channels if in_channels > 1 else 1
        out_ch = 4 * groups

        self.fixed_conv = nn.Conv2d(in_channels, out_ch, 2, stride=stride,
                                    padding=0, bias=False, groups=groups)
        w = templates.unsqueeze(1).repeat(groups, 1, 1, 1)
        self.fixed_conv.weight = nn.Parameter(w, requires_grad=False)

        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_ch)

        self.activation = activation
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'leaky':
            self.act = nn.LeakyReLU(0.1)
        elif activation == 'abs':
            self.act = None  # обрабатываем в forward
        else:
            raise ValueError(activation)

        self.features = nn.Sequential(
            nn.Conv2d(out_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fixed_conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation == 'abs':
            x = torch.abs(x)
        else:
            x = self.act(x)
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


# ============================================================
# ДАННЫЕ
# ============================================================

def get_loaders(dataset_name, batch_size=128):
    tf = transforms.Compose([transforms.ToTensor()])
    if dataset_name == "MNIST":
        cls, root = datasets.MNIST, "./data"
    elif dataset_name == "FashionMNIST":
        cls, root = datasets.FashionMNIST, "./data"
    else:
        raise ValueError(dataset_name)

    train_ds = cls(root=root, train=True,  download=True, transform=tf)
    test_ds  = cls(root=root, train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1000, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ============================================================
# ОБУЧЕНИЕ
# ============================================================

def run(dataset_name, cfg, epochs=15, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loaders(dataset_name)
    model = BasisNet(
        in_channels=1,
        templates=cfg["templates"],
        use_bn=cfg["use_bn"],
        activation=cfg["activation"],
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=lr)
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
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = [
        ("1. Baseline (ReLU, шаблоны 1/-1)",
            dict(templates=TEMPLATES,      use_bn=False, activation='relu')),
        ("2. +BN после fixed_conv",
            dict(templates=TEMPLATES,      use_bn=True,  activation='relu')),
        ("3. +abs вместо ReLU",
            dict(templates=TEMPLATES,      use_bn=False, activation='abs')),
        ("4. +шаблоны /2",
            dict(templates=TEMPLATES_NORM, use_bn=False, activation='relu')),
        ("5. +LeakyReLU (0.1)",
            dict(templates=TEMPLATES,      use_bn=False, activation='leaky')),
        ("6. Всё сразу: /2 + BN + abs",
            dict(templates=TEMPLATES_NORM, use_bn=True,  activation='abs')),
    ]

    datasets_to_run = ["MNIST", "FashionMNIST"]
    summary = {}

    for ds in datasets_to_run:
        print(f"\n{'='*78}\nНАБОР: {ds}  (15 эпох, Cosine, без нормализации данных)\n{'='*78}")
        rows = []
        for label, cfg in configs:
            print(f"\n>>> {label}")
            acc, t = run(ds, cfg=cfg, epochs=15)
            print(f"    Точность: {acc:.2f}%  |  Время: {t:.1f}с")
            rows.append((label, acc, t))
        summary[ds] = rows

    # --- Сводка ---
    print("\n\n" + "=" * 82)
    print("СВОДКА".center(82))
    print("=" * 82)
    for ds in datasets_to_run:
        print(f"\n### {ds} ###")
        print(f"{'Конфигурация':<42}{'Точность':>12}{'Время':>10}")
        print("-" * 64)
        for label, acc, t in summary[ds]:
            print(f"{label:<42}{acc:>11.2f}%{t:>9.1f}с")