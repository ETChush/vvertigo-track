# -*- coding: utf-8 -*-
"""外观编码器。

第一版用 ImageNet 预训练的 ResNet18 取倒数第二层 512 维特征。它不是 ReID 专用
模型（没有在行人数据集上微调过），但在「同一段素材内做同一性比对」这个任务上
够用：光照和机位固定，主要外观差异来自服装颜色和轮廓。

接口留成可替换的：只要 embed() 返回 L2 归一化向量，换成 OSNet / CLIP / DINOv2
都不影响下游。
"""
from __future__ import annotations

import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ResNetEmbedder:
    name = "resnet18"

    def __init__(self, device="cuda", size: tuple[int, int] = (128, 256)):
        import torch
        from torchvision.models import resnet18, ResNet18_Weights

        self.torch = torch
        self.size = size                       # (w, h)
        self.device = device if torch.cuda.is_available() else "cpu"
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        self.model = m.eval().to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()

    def embed(self, frame: np.ndarray, boxes: list[tuple[float, float, float, float]],
              pad: float = 0.04) -> np.ndarray:
        """frame: BGR；boxes: (x1,y1,x2,y2)。返回 (N, 512) float32，已归一化。"""
        import cv2
        import torch

        if not boxes:
            return np.zeros((0, 512), dtype=np.float32)
        H, W = frame.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes:
            w, h = x2 - x1, y2 - y1
            px, py = w * pad, h * pad
            a = max(0, int(x1 - px)); b = max(0, int(y1 - py))
            c = min(W, int(x2 + px)); d = min(H, int(y2 + py))
            if c - a < 4 or d - b < 4:
                crops.append(np.zeros((self.size[1], self.size[0], 3), np.uint8))
                continue
            crops.append(cv2.resize(frame[b:d, a:c], self.size,
                                    interpolation=cv2.INTER_AREA))
        batch = np.stack(crops).astype(np.float32)[:, :, ::-1] / 255.0
        batch = (batch - _MEAN) / _STD
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        if self.device == "cuda":
            x = x.half()
        with torch.no_grad():
            f = self.model(x)
        f = f.float().cpu().numpy()
        n = np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
        return (f / n).astype(np.float32)


class HistEmbedder:
    """无 GPU 时的兜底：HSV 3D 直方图。弱，但不需要任何权重。"""
    name = "hsv"

    def embed(self, frame, boxes, pad: float = 0.04) -> np.ndarray:
        import cv2

        if not boxes:
            return np.zeros((0, 512), dtype=np.float32)
        H, W = frame.shape[:2]
        out = np.zeros((len(boxes), 512), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            a = max(0, int(x1)); b = max(0, int(y1))
            c = min(W, int(x2)); d = min(H, int(y2))
            if c - a < 4 or d - b < 4:
                continue
            hsv = cv2.cvtColor(frame[b:d, a:c], cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            v = h.flatten().astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            out[i] = v
        return out


def make_embedder(kind: str = "resnet18"):
    if kind == "hsv":
        return HistEmbedder()
    try:
        return ResNetEmbedder()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] resnet 不可用（{exc}），退回 HSV 直方图")
        return HistEmbedder()


def lowbody_desc(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """下半身颜色描述子（短裤下沿到靴子）。

    这批舞者上半身几乎同款白衣，能分开人的是裤子/靴子——5 个人里有人穿白色
    过膝靴、有人穿黑色厚底靴。整身平均特征会把这种差异average掉，所以单独取
    下半身做一路证据，用来在"重新绑定"时否决明显不是同一个人的候选。
    """
    import cv2

    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    h = y2 - y1
    a, b = int(max(0, x1)), int(max(0, y1 + h * 0.68))
    c, d = int(min(W, x2)), int(min(H, y1 + h * 1.02))
    if c - a < 6 or d - b < 8:
        return None
    hsv = cv2.cvtColor(frame[b:d, a:c], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256]).flatten()
    total = hist.sum()
    if total <= 0:
        return None
    v = np.sqrt(hist / total).astype(np.float32)     # Bhattacharyya 形式
    return v / (np.linalg.norm(v) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
