"""
export_onnx.py
==============
Обучает лучшую модель и экспортирует её в ONNX (fp32 и int8).
int8-квантование ограничено Linear-слоями — WASM не поддерживает ConvInteger.

Запуск:
    python export_onnx.py
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

TEMPLATES = torch.tensor([
    [[ 1,  1], [ 1,  1]],
    [[ 1,  1], [-1, -1]],
    [[ 1, -1], [ 1, -1]],
    [[ 1, -1], [-1,  1]],
], dtype=torch.float32)


class BasisNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.fixed_conv = nn.Conv2d(in_channels, 4, 2, stride=2,
                                    padding=0, bias=False)
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


def train(epochs=15, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = BasisNet().to(device)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        if ep % 5 == 0 or ep == epochs:
            print(f"  эпоха {ep:>2}/{epochs} | loss = {total/len(train_loader):.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    print(f"Точность на тесте: {100 * correct / len(test_ds):.2f}%")

    return model.cpu()


def export_onnx(model, path="web/model.onnx"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 1, 28, 28)

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"ONNX сохранён: {path}  ({os.path.getsize(path) / 1024:.1f} KB)")
    return path


def quantize_onnx(src_path, dst_path="web/model_int8.onnx"):
    """
    Квантует только Linear-слои (MatMul/Gemm).
    Conv-слои остаются в fp32, потому что ONNX Runtime Web (WASM)
    не поддерживает оператор ConvInteger.

    Уменьшение размера: ~1.5-2x (не 4x, но зато работает в браузере).
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("onnxruntime.quantization не найден. Установите: pip install onnxruntime")
        return None

    # --- Попытка 1: int8 только для MatMul/Gemm ---
    try:
        quantize_dynamic(
            src_path,
            dst_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            extra_options={
                "WeightSymmetric": True,
                "MatMulConstBOnly": True,
            },
        )
        print(f"int8 ONNX сохранён (MatMul/Gemm): {dst_path}  "
              f"({os.path.getsize(dst_path) / 1024:.1f} KB)")
        return dst_path
    except Exception as e:
        print(f"int8 (QInt8) не удался: {e}")
        print("Пробуем uint8...")

    # --- Попытка 2: uint8 целиком (WASM поддерживает лучше) ---
    try:
        quantize_dynamic(
            src_path,
            dst_path,
            weight_type=QuantType.QUInt8,
        )
        print(f"uint8 ONNX сохранён: {dst_path}  "
              f"({os.path.getsize(dst_path) / 1024:.1f} KB)")
        return dst_path
    except Exception as e:
        print(f"uint8 тоже не удался: {e}")
        print("Оставляем только fp32-версию.")
        return None


def save_reference(model, num_samples=10):
    """Сохраняем эталонные входы/выходы для проверки в браузере."""
    model.eval()
    tf = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tf)

    inputs, logits, labels = [], [], []
    for i in range(num_samples):
        x, y = test_ds[i]
        x = x.unsqueeze(0)
        with torch.no_grad():
            out = model(x)
        inputs.append(x.flatten().tolist())
        logits.append(out.flatten().tolist())
        labels.append(y)

    data = {
        "inputs": inputs,
        "logits_fp32": logits,
        "labels": labels,
    }
    os.makedirs("web", exist_ok=True)
    with open("web/reference.json", "w") as f:
        json.dump(data, f)
    print(f"Эталон сохранён: web/reference.json ({num_samples} примеров)")


if __name__ == "__main__":
    model = train(epochs=15)
    onnx_path = export_onnx(model)
    quantize_onnx(onnx_path)
    save_reference(model)
    print("\nГотово. Дальше: python -m http.server 8000 --directory web")