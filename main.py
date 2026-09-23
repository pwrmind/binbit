"""
router_cascade.py
=================
Проверяем, имеет ли смысл MoE-подход поверх нашей архитектуры.

Компоненты:
    1. Три независимых эксперта (MNIST / FashionMNIST / KMNIST)
    2. Domain router: классификатор домена на объединённом датасете
    3. Каскад: router → соответствующий эксперт
    4. Baseline: единая модель на 30 классах

Всё с одинаковыми гиперпараметрами: 15 эпох, Cosine, ToTensor без нормализации.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
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
# МОДЕЛЬ
# ============================================================

class BasisNet(nn.Module):
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


# ============================================================
# ДАННЫЕ
# ============================================================

DOMAIN_NAMES = ["MNIST", "FashionMNIST", "KMNIST"]


def get_raw_dataset(name, train):
    tf = transforms.Compose([transforms.ToTensor()])
    if name == "MNIST":
        cls, root = datasets.MNIST, "./data"
    elif name == "FashionMNIST":
        cls, root = datasets.FashionMNIST, "./data"
    elif name == "KMNIST":
        cls, root = datasets.KMNIST, "./data"
    else:
        raise ValueError(name)
    return cls(root=root, train=train, download=True, transform=tf)


class DomainLabeled(torch.utils.data.Dataset):
    """Обёртка, добавляющая метку домена к каждому примеру."""
    def __init__(self, dataset, domain_id):
        self.dataset = dataset
        self.domain_id = domain_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return x, self.domain_id


class GlobalLabeled(torch.utils.data.Dataset):
    """Обёртка для единой модели: метка = domain_id * 10 + original_class."""
    def __init__(self, dataset, domain_id):
        self.dataset = dataset
        self.offset = domain_id * 10

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y + self.offset


def get_domain_loaders(batch_size=128):
    """Объединённый датасет с метками доменов 0/1/2."""
    train_parts = [DomainLabeled(get_raw_dataset(n, True), i)
                   for i, n in enumerate(DOMAIN_NAMES)]
    test_parts = [DomainLabeled(get_raw_dataset(n, False), i)
                  for i, n in enumerate(DOMAIN_NAMES)]
    train_ds = ConcatDataset(train_parts)
    test_ds = ConcatDataset(test_parts)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True),
            DataLoader(test_ds, batch_size=1000, shuffle=False,
                       num_workers=2, pin_memory=True))


def get_global_loaders(batch_size=128):
    """Объединённый датасет с метками 0..29."""
    train_parts = [GlobalLabeled(get_raw_dataset(n, True), i)
                   for i, n in enumerate(DOMAIN_NAMES)]
    test_parts = [GlobalLabeled(get_raw_dataset(n, False), i)
                  for i, n in enumerate(DOMAIN_NAMES)]
    train_ds = ConcatDataset(train_parts)
    test_ds = ConcatDataset(test_parts)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True),
            DataLoader(test_ds, batch_size=1000, shuffle=False,
                       num_workers=2, pin_memory=True))


# ============================================================
# ОБУЧЕНИЕ: ОДИНОЧНАЯ МОДЕЛЬ НА N КЛАССОВ
# ============================================================

def train_model(train_loader, num_classes, epochs=15, lr=1e-3, seed=42, label=""):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasisNet(num_classes=num_classes).to(device)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()

    for ep in range(epochs):
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
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            print(f"    {label} эпоха {ep+1:>2}/{epochs} | loss = {total_loss/len(train_loader):.4f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t = time.time() - t0
    return model, t


def eval_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return 100 * correct / total


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    EPOCHS = 15
    SEED = 42

    # ---------- 1. Три эксперта ----------
    print(f"\n{'='*72}\n1. ТРИ НЕЗАВИСИМЫХ ЭКСПЕРТА\n{'='*72}")
    experts = []
    expert_accs = []
    for i, name in enumerate(DOMAIN_NAMES):
        print(f"\n  >>> Эксперт {i+1}: {name}")
        train_ds = get_raw_dataset(name, train=True)
        test_ds  = get_raw_dataset(name, train=False)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                                  num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False,
                                  num_workers=2, pin_memory=True)
        model, t = train_model(train_loader, num_classes=10,
                               epochs=EPOCHS, seed=SEED, label=name)
        acc = eval_accuracy(model, test_loader, device)
        print(f"    Точность {name}: {acc:.2f}%  |  время: {t:.1f}с")
        experts.append(model)
        expert_accs.append(acc)

    # ---------- 2. Domain router ----------
    print(f"\n{'='*72}\n2. DOMAIN ROUTER (3 класса)\n{'='*72}")
    router_train, router_test = get_domain_loaders()
    router, t_router = train_model(router_train, num_classes=3,
                                    epochs=EPOCHS, seed=SEED, label="router")
    router_acc = eval_accuracy(router, router_test, device)
    print(f"    Точность router: {router_acc:.2f}%  |  время: {t_router:.1f}с")

    # ---------- 3. Каскад ----------
    print(f"\n{'='*72}\n3. КАСКАД: router → expert\n{'='*72}")
    cascade_correct_total = 0
    cascade_total = 0
    per_domain_cascade = []

    for i, name in enumerate(DOMAIN_NAMES):
        test_ds = get_raw_dataset(name, train=False)
        test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False,
                                 num_workers=2, pin_memory=True)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                # Router выбирает домен
                domain_pred = router(x).argmax(1)
                # Эксперт работает по своему домену
                expert_pred = experts[i](x).argmax(1)
                # Правильно, если router попал в этот домен И эксперт угадал класс
                correct += ((domain_pred == i) & (expert_pred == y)).sum().item()
                total += y.numel()
        acc = 100 * correct / total
        per_domain_cascade.append(acc)
        cascade_correct_total += correct
        cascade_total += total
        print(f"    {name:<15} точность каскада: {acc:.2f}%")

    cascade_overall = 100 * cascade_correct_total / cascade_total
    print(f"    Средняя по всем доменам: {cascade_overall:.2f}%")

    # ---------- 4. Единая модель на 30 классах ----------
    print(f"\n{'='*72}\n4. ЕДИНАЯ МОДЕЛЬ на 30 классах (baseline)\n{'='*72}")
    global_train, global_test = get_global_loaders()
    single_model, t_single = train_model(global_train, num_classes=30,
                                          epochs=EPOCHS, seed=SEED, label="single30")
    single_acc = eval_accuracy(single_model, global_test, device)
    print(f"    Точность единой модели: {single_acc:.2f}%  |  время: {t_single:.1f}с")

    # ---------- Сводка ----------
    print("\n\n" + "=" * 82)
    print("СВОДКА".center(82))
    print("=" * 82)
    print(f"\n{'Метод':<36}{'MNIST':>10}{'Fashion':>10}{'KMNIST':>10}{'Среднее':>10}")
    print("-" * 82)
    print(f"{'Три независимых эксперта':<36}"
          f"{expert_accs[0]:>9.2f}%{expert_accs[1]:>9.2f}%{expert_accs[2]:>9.2f}%"
          f"{sum(expert_accs)/3:>9.2f}%")
    print(f"{'Каскад (router → expert)':<36}"
          f"{per_domain_cascade[0]:>9.2f}%{per_domain_cascade[1]:>9.2f}%{per_domain_cascade[2]:>9.2f}%"
          f"{cascade_overall:>9.2f}%")
    print(f"{'Единая модель (30 классов)':<36}"
          f"{'—':>10}{'—':>10}{'—':>10}{single_acc:>9.2f}%")
    print("=" * 82)

    print(f"\nТочность router: {router_acc:.2f}%")
    print(f"Разница эксперты − каскад: {sum(expert_accs)/3 - cascade_overall:.2f}% (это цена роутинга)")