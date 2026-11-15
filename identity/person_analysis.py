# -*- coding: utf-8 -*-
"""InsightFace 2.1 PersonAnalysis 适配器（人体 ReID + 人脸，一次前向）。

为什么值得接入：自家 ResNet18 的整身特征在同一人 0.767 / 不同人 0.91 之间完全
重叠；而 cheetah 的 reid_feature 实测同一人 0.76–0.85、不同人 0.55–0.67，
是唯一能用的"整体外观"信号。它同时给出人脸，可以和我们的 SFace 互为校验。

注意：它自带人物检测器（cheetah_s 输入 320×320），框和 YOLO 不会重合，所以
输出必须通过 IoU 关联回 Local Track，关联不可靠时宁可不给证据。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PersonObs:
    bbox: tuple[float, float, float, float]
    score: float
    reid: np.ndarray | None
    face: np.ndarray | None
    face_valid: bool


class PersonAnalyzer:
    def __init__(self, name: str = "cheetah_s", providers=None):
        from insightface.app import PersonAnalysis   # 延迟导入

        # 默认优先 GPU。CPU 版的 onnxruntime 在这个模型上慢到不可用：
        # 10486 帧的素材光人体特征就要 23 分钟，换 CUDA 后是分钟级。
        if providers is None:
            try:
                import onnxruntime as ort
                av = ort.get_available_providers()
                providers = (("CUDAExecutionProvider", "CPUExecutionProvider")
                             if "CUDAExecutionProvider" in av else ("CPUExecutionProvider",))
            except Exception:      # noqa: BLE001
                providers = ("CPUExecutionProvider",)
        self.app = PersonAnalysis(name=name, providers=list(providers))
        self.provider = list(providers)[0]
        self.name = f"insightface:{name}"

    def get(self, frame: np.ndarray) -> list[PersonObs]:
        out: list[PersonObs] = []
        try:
            obs = self.app.get(frame)
        except Exception as exc:      # noqa: BLE001
            print(f"[warn] PersonAnalysis 失败: {exc}")
            return out
        for o in obs:
            bb = getattr(o, "body_bbox", None)
            if bb is None:
                continue
            reid = getattr(o, "reid_feature", None)
            if reid is not None:
                reid = np.asarray(reid, dtype=np.float32).ravel()
                reid = reid / (np.linalg.norm(reid) + 1e-9)
            f = getattr(o, "face", None)
            feat = None
            valid = False
            if f is not None:
                emb = getattr(f, "normed_embedding", None) or getattr(f, "embedding", None)
                if emb is not None:
                    feat = np.asarray(emb, dtype=np.float32).ravel()
                    feat = feat / (np.linalg.norm(feat) + 1e-9)
                    valid = bool(getattr(o, "_association_valid", True))
            out.append(PersonObs(bbox=(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])),
                                 score=float(getattr(o, "det_score", 0.0)),
                                 reid=reid, face=feat, face_valid=valid))
        return out

    def close(self) -> None:
        try:
            self.app.close()
        except Exception:      # noqa: BLE001
            pass


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def associate(persons: list[PersonObs], boxes: list[tuple], min_iou: float = 0.35,
              max_center_shift: float = 0.35) -> dict[int, PersonObs]:
    """把 PersonAnalysis 的框关联到本地轨迹的框；取 IoU 最高者，且必须明显优于次优。"""
    out: dict[int, PersonObs] = {}
    for key, b in boxes:
        best, best_iou, second = None, 0.0, 0.0
        for p in persons:
            v = iou(b, p.bbox)
            if v > best_iou:
                best, second, best_iou = p, best_iou, v
            elif v > second:
                second = v
        if best is None:
            continue
        if best_iou >= min_iou and best_iou >= second * 1.6:
            out[key] = best
    return out
