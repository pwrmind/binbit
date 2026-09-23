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
# УНИВЕРСАЛЬНАЯ СЕТЬ
# ============================================================

class Net(nn.Module):
    def __init__(self, in_channels, n_templates=4, templates=None,
                 stride=2, activation='none', trainable=False):
        """
        in_channels : 1 (MNIST-семейство) или 3 (CIFAR-10)
        n_templates : сколько уникальных фильтров 2x2
        templates   : тензор (n_templates, 2, 2) — используется, если trainable=False
        activation  : 'none' | 'relu' | 'leaky'
        trainable   : если True — conv обучаемый, templates игнорируется
        """
        super().__init__()
        groups = in_channels if in_channels > 1 else 1
        out_ch = n_templates * groups

        self.conv = nn.Conv2d(in_channels, out_ch, kernel_size=2, stride=stride,
                              padding=0, bias=False, groups=groups)

        if not trainable and templates is not None:
            w = templates.unsqueeze(1).repeat(groups, 1, 1, 1)  # (out_ch, 1, 2, 2)
            self.conv.weight = nn.Parameter(w, requires_grad=False)

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'leaky':
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = nn.Identity()

        self.body = nn.Sequential(
            nn.Conv2d(out_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.body(x)
        return self.classifier(torch.flatten(x, 1))


# ============================================================
# ДАННЫЕ
# ============================================================

def get_datasets(name):
    tf = transforms.Compose([transforms.ToTensor()])
    if name == "MNIST":
        tr = datasets.MNIST('./data', train=True,  download=True, transform=tf)
        te = datasets.MNIST('./data', train=False, download=True, transform=tf)
    elif name == "FashionMNIST":
        tr = datasets.FashionMNIST('./data', train=True,  download=True, transform=tf)
        te = datasets.FashionMNIST('./data', train=False, download=True, transform=tf)
    elif name == "KMNIST":
        tr = datasets.KMNIST('./data', train=True,  download=True, transform=tf)
        te = datasets.KMNIST('./data', train=False, download=True, transform=tf)
    elif name == "CIFAR10":
        tr = datasets.CIFAR10('./data_cifar', train=True,  download=True, transform=tf)
        te = datasets.CIFAR10('./data_cifar', train=False, download=True, transform=tf)
    else:
        raise ValueError(name)
    return tr, te


# ============================================================
# ОБУЧЕНИЕ
# ============================================================

def train_and_eval(cfg, dataset_name, epochs, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = get_datasets(dataset_name)
    in_ch = 3 if dataset_name == "CIFAR10" else 1

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = Net(
        in_channels=in_ch,
        n_templates=cfg["n_templates"],
        templates=cfg["templates"],
        stride=cfg["stride"],
        activation=cfg["activation"],
        trainable=cfg["trainable"],
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=0.001)
    crit = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t = time.time() - t0

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item()

    acc = 100 * correct / len(test_ds)
    n_params = sum(p.numel() for p in trainable)
    return acc, t, n_params


# ============================================================
# КОНФИГУРАЦИИ
# ============================================================

CONFIGS = [
    # 1) Новая база, которую мы вывели из предыдущих экспериментов
    dict(name="4 фикс + ReLU",        n_templates=4, templates=TEMPLATES_4,     stride=2, activation='relu',  trainable=False),
    # 2) Обучаемый слой с тем же бюджетом каналов
    dict(name="4 обуч. + ReLU",       n_templates=4, templates=None,            stride=2, activation='relu',  trainable=True),
    # 3) LeakyReLU — проверить отрицательную ветку
    dict(name="4 фикс + LeakyReLU",   n_templates=4, templates=TEMPLATES_4,     stride=2, activation='leaky', trainable=False),
    # 4) Контроль: 8 фиксированных + ReLU (наш прошлый «условный чемпион»)
    dict(name="8 фикс + ReLU",        n_templates=8, templates=TEMPLATES_8,     stride=2, activation='relu',  trainable=False),
    # 5) Контроль: 4×2 дубликат + ReLU (та же перепараметризация)
    dict(name="4×2 фикс + ReLU",      n_templates=8, templates=TEMPLATES_4_DUP, stride=2, activation='relu',  trainable=False),
]


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    plan = [
        ("MNIST",        5),
        ("FashionMNIST", 5),
        ("KMNIST",       5),
        ("CIFAR10",      15),
    ]

    summary = {}
    for ds_name, epochs in plan:
        print(f"\n{'='*78}\nНАБОР: {ds_name}  (эпох: {epochs})\n{'='*78}")
        rows = []
        for cfg in CONFIGS:
            acc, t, n_params = train_and_eval(cfg, ds_name, epochs)
            print(f"  {cfg['name']:<22} точн={acc:6.2f}%  время={t:5.1f}с  параметров={n_params:>8,}")
            rows.append((cfg['name'], acc, t, n_params))
        summary[ds_name] = rows

    # Итог
    print("\n\n" + "=" * 90)
    print("СВОДНАЯ ТАБЛИЦА".center(90))
    print("=" * 90)
    header = f"{'Конфигурация':<24}" + "".join(f"{ds:>16}" for ds, _ in plan)
    print(header)
    print("-" * 90)
    for i, cfg in enumerate(CONFIGS):
        line = f"{cfg['name']:<24}"
        for ds_name, _ in plan:
            acc = summary[ds_name][i][1]
            line += f"{acc:>15.2f}%"
        print(line)
    print("=" * 90)