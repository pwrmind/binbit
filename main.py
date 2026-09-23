"""
multitask_experiment.py
=======================
Проверяем, можно ли переиспользовать фиксированный геометрический базис
для нескольких задач одновременно.

Режимы:
    1. Independent        : 3 отдельные BasisNet, обучены независимо (baseline)
    2. Shared trunk       : общий фикс. базис + общий ствол + 3 головы
    3. Shared basis only  : общий фикс. базис + 3 отдельных ствола + 3 головы

Задачи: MNIST, FashionMNIST, KMNIST. Все 28x28, 10 классов.
15 эпох, CosineAnnealingLR, ToTensor без нормализации.
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
# ФИКСИРОВАННЫЙ БАЗИС
# ============================================================

TEMPLATES = torch.tensor([
    [[ 1,  1], [ 1,  1]],
    [[ 1,  1], [-1, -1]],
    [[ 1, -1], [ 1, -1]],
    [[ 1, -1], [-1,  1]],
], dtype=torch.float32)


# ============================================================
# МОДЕЛИ
# ============================================================

class BasisNet(nn.Module):
    """Одиночная модель — для baseline."""
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.fixed_conv = nn.Conv2d(in_channels, 4, 2, stride=2, padding=0, bias=False)
        w = TEMPLATES.unsqueeze(1)
        self.fixed_conv.weight = nn.Parameter(w, requires_grad=False)
        self.act = nn.ReLU()
        self.trunk = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.act(self.fixed_conv(x))
        x = self.trunk(x)
        return self.classifier(torch.flatten(x, 1))


class MultiTaskNet(nn.Module):
    """
    Общий фиксированный базис + (общий или раздельный) ствол + N голов.
    task_id передаётся в forward, чтобы выбрать нужный ствол/голову.
    """
    def __init__(self, num_tasks=3, in_channels=1, num_classes=10, share_trunk=True):
        super().__init__()
        self.share_trunk = share_trunk
        self.num_tasks = num_tasks

        # --- Фиксированный базис (всегда общий) ---
        self.fixed_conv = nn.Conv2d(in_channels, 4, 2, stride=2, padding=0, bias=False)
        w = TEMPLATES.unsqueeze(1)
        self.fixed_conv.weight = nn.Parameter(w, requires_grad=False)
        self.act = nn.ReLU()

        def make_trunk():
            return nn.Sequential(
                nn.Conv2d(4, 64, 3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

        if share_trunk:
            self.trunk = make_trunk()
        else:
            self.trunks = nn.ModuleList([make_trunk() for _ in range(num_tasks)])

        self.heads = nn.ModuleList([nn.Linear(128, num_classes) for _ in range(num_tasks)])

    def forward(self, x, task_id):
        x = self.act(self.fixed_conv(x))
        if self.share_trunk:
            x = self.trunk(x)
        else:
            x = self.trunks[task_id](x)
        x = torch.flatten(x, 1)
        return self.heads[task_id](x)


# ============================================================
# ДАННЫЕ
# ============================================================

def get_loaders(name, batch_size=128):
    tf = transforms.Compose([transforms.ToTensor()])
    if name == "MNIST":
        cls, root = datasets.MNIST, "./data"
    elif name == "FashionMNIST":
        cls, root = datasets.FashionMNIST, "./data"
    elif name == "KMNIST":
        cls, root = datasets.KMNIST, "./data"
    else:
        raise ValueError(name)

    tr = cls(root=root, train=True,  download=True, transform=tf)
    te = cls(root=root, train=False, download=True, transform=tf)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True),
            DataLoader(te, batch_size=1000, shuffle=False,
                       num_workers=2, pin_memory=True))


# ============================================================
# ОБУЧЕНИЕ ОДИНОЧНОЙ МОДЕЛИ (BASELINE)
# ============================================================

def train_single(dataset_name, epochs=15, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loaders(dataset_name)
    model = BasisNet().to(device)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item()
    acc = 100 * correct / len(test_loader.dataset)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return acc, n_params


# ============================================================
# ОБУЧЕНИЕ МУЛЬТИЗАДАЧНОЙ МОДЕЛИ
# ============================================================

def train_multitask(dataset_names, share_trunk, epochs=15, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = [get_loaders(name) for name in dataset_names]
    train_loaders = [l[0] for l in loaders]
    test_loaders  = [l[1] for l in loaders]

    model = MultiTaskNet(num_tasks=len(dataset_names),
                         share_trunk=share_trunk).to(device)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    # Итераторы на все задачи; по одному батчу на задачу за шаг
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch

    iters = [cycle(l) for l in train_loaders]
    steps_per_epoch = min(len(l) for l in train_loaders)

    for ep in range(epochs):
        model.train()
        for _ in range(steps_per_epoch):
            opt.zero_grad()
            total_loss = 0.0
            for task_id in range(len(dataset_names)):
                x, y = next(iters[task_id])
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                out = model(x, task_id=task_id)
                total_loss = total_loss + crit(out, y)
            total_loss.backward()
            opt.step()
        sched.step()

    # Оценка по каждой задаче
    accs = []
    model.eval()
    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            correct = 0
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x, task_id=task_id)
                correct += (out.argmax(1) == y).sum().item()
            accs.append(100 * correct / len(test_loader.dataset))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return accs, n_params


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset_names = ["MNIST", "FashionMNIST", "KMNIST"]
    epochs = 15

    # ---------- 1. Независимые модели ----------
    print(f"\n{'='*72}\n1. INDEPENDENT: 3 отдельные модели\n{'='*72}")
    indep_accs = []
    indep_params_total = 0
    for name in dataset_names:
        t0 = time.time()
        acc, n_params = train_single(name, epochs=epochs)
        t = time.time() - t0
        print(f"  {name:<15} точн={acc:6.2f}%  параметров={n_params:>8,}  время={t:5.1f}с")
        indep_accs.append(acc)
        indep_params_total += n_params

    # ---------- 2. Общий ствол ----------
    print(f"\n{'='*72}\n2. SHARED TRUNK: общий базис + общий ствол + 3 головы\n{'='*72}")
    t0 = time.time()
    shared_trunk_accs, shared_trunk_params = train_multitask(
        dataset_names, share_trunk=True, epochs=epochs)
    t = time.time() - t0
    for name, acc in zip(dataset_names, shared_trunk_accs):
        print(f"  {name:<15} точн={acc:6.2f}%")
    print(f"  Всего параметров: {shared_trunk_params:,}  |  время: {t:.1f}с")

    # ---------- 3. Только базис общий ----------
    print(f"\n{'='*72}\n3. SHARED BASIS ONLY: общий базис + 3 ствола + 3 головы\n{'='*72}")
    t0 = time.time()
    shared_basis_accs, shared_basis_params = train_multitask(
        dataset_names, share_trunk=False, epochs=epochs)
    t = time.time() - t0
    for name, acc in zip(dataset_names, shared_basis_accs):
        print(f"  {name:<15} точн={acc:6.2f}%")
    print(f"  Всего параметров: {shared_basis_params:,}  |  время: {t:.1f}с")

    # ---------- Сводка ----------
    print("\n\n" + "=" * 88)
    print("СВОДКА".center(88))
    print("=" * 88)
    header = f"{'Режим':<26}{'MNIST':>10}{'Fashion':>10}{'KMNIST':>10}{'Параметры':>14}{'Ср.точн':>10}"
    print(header)
    print("-" * 88)

    def row(label, accs, params):
        avg = sum(accs) / len(accs)
        return (f"{label:<26}{accs[0]:>9.2f}%{accs[1]:>9.2f}%{accs[2]:>9.2f}%"
                f"{params:>13,}{avg:>9.2f}%")

    print(row("Independent (3 модели)", indep_accs, indep_params_total))
    print(row("Shared trunk", shared_trunk_accs, shared_trunk_params))
    print(row("Shared basis only", shared_basis_accs, shared_basis_params))
    print("=" * 88)