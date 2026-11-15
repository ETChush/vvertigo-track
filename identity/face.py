# -*- coding: utf-8 -*-
"""人脸编码器（YuNet 检测 + SFace 识别，纯 cv2，无需编译 insightface）。

为什么用人脸：本素材上人体外观几乎没有区分度（同一人 0.767 / 不同人 0.91），
而人脸在同一人 0.80 / 不同人最高 0.58——这是唯一能把 5 名舞者分开的模态。

为什么不用逐帧人脸分数做决策：46px 的人脸在侧脸、低头、运动模糊时相似度会掉到
0.3，单帧证据不可靠。所以只把「一条局部轨迹上多帧人脸的平均特征」当作证据，
单帧分数只用于质量过滤。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

YUNET = "work/models/yunet.onnx"
SFACE = "work/models/sface.onnx"


@dataclass
class FaceObs:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    feat: np.ndarray

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


class FaceEncoder:
    def __init__(self, yunet: str = YUNET, sface: str = SFACE,
                 min_size: float = 32.0, score_thresh: float = 0.6):
        import cv2

        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(yunet, "", (1920, 1080), score_thresh, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(sface, "")
        self.min_size = min_size

        # SFace 走 GPU。检测器保持 cv2：YuNet 在 onnxruntime 上实测反而慢一倍
        # （每次调用都要 640x640 预处理 + CPU 端的 NMS，开销远大于 3ms 的推理，
        # 而 cv2 的实现全在 C++ 里）。识别是 112x112 的小前向，GPU 收益明确：
        # 4.16ms → 0.91ms，特征完全一致（cos > 0.999）。
        # 默认走 CPU。单独测时 GPU 是 0.91ms（CPU 4.16ms），但放进整条流水线后
        # 反而慢 5 倍（人脸 122s → 390s）：它和 torch 抢同一个 CUDA context，
        # 每次只传 112x112 的小张量，传输+同步的开销远大于计算收益。
        # 只在确定不会有 torch 抢卡时用 `--face-gpu` 打开。
        self.sf_onnx = None
        if os.environ.get("VVERTIGO_SFACE_GPU") == "1":
            try:
                from identity.face_onnx import OnnxSFace

                self.sf_onnx = OnnxSFace(sface)
            except Exception:                          # noqa: BLE001
                self.sf_onnx = None

    def _feature(self, aligned):
        """提特征。优先 GPU，失败回退 cv2。"""
        if self.sf_onnx is not None:
            try:
                return self.sf_onnx.feature(aligned)
            except Exception:                          # noqa: BLE001
                self.sf_onnx = None
        v = self.rec.feature(aligned).flatten().astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def embed_aligned(self, img: np.ndarray) -> np.ndarray:
        """输入已对齐的 112x112 人脸（或任意人脸图），直接出特征。"""
        import cv2

        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
        v = self.rec.feature(img).flatten().astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def embed_file(self, path: str) -> np.ndarray:
        import cv2

        img = None
        try:
            buf = np.fromfile(str(path), dtype=np.uint8)   # 中文路径 imread 会失败
            if buf.size:
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except OSError:
            img = None
        if img is None:
            raise SystemExit(f"读不到人脸图: {path}")
        return self.embed_aligned(img)

    def embed(self, frame: np.ndarray) -> list[FaceObs]:
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(frame)
        out: list[FaceObs] = []
        if faces is None:
            return out
        for f in faces:
            x, y, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
            if fh < self.min_size:
                continue
            try:
                aligned = self.rec.alignCrop(frame, f)
                v = self._feature(aligned)
            except Exception:      # noqa: BLE001
                continue
            out.append(FaceObs(x, y, x + fw, y + fh, float(f[-1]), v))
        return out


    def embed_head(self, frame: np.ndarray, box, top: float = 0.42,
                   up: int = 2, max_side: int = 640) -> list[FaceObs]:
        """只在一个人体框的"头部区域"里找人脸，坐标映射回原图。

        全帧降采样会把人脸压没：4K 素材降到 1280 高时，舞者的脸只剩 60px 左右，
        识别相似度会掉到分不开人的程度。裁出头部再放大 2 倍，等于给每张脸足够的像素，
        而且只处理人头所在的那一小块，比整帧检测更快。
        """
        import cv2

        x1, y1, x2, y2 = [float(v) for v in box]
        h = y2 - y1
        hy2 = y1 + h * top
        X1, Y1 = max(0, int(x1)), max(0, int(y1))
        X2, Y2 = int(x2), int(min(frame.shape[0], hy2))
        if X2 - X1 < 12 or Y2 - Y1 < 12:
            return []
        sub = frame[Y1:Y2, X1:X2]
        # 保持"按 up 倍放大"的原逻辑。试过统一 letterbox 到 640x640 来避开
        # setInputSize 重建，速度只快 20%，但框位置有微小偏移，锚点从 145 掉到
        # 123、覆盖掉 7 个点——不值。精度优先。
        if up != 1:
            sub = cv2.resize(sub, (sub.shape[1] * up, sub.shape[0] * up),
                             interpolation=cv2.INTER_CUBIC)
        self.det.setInputSize((sub.shape[1], sub.shape[0]))
        n, faces = self.det.detect(sub)
        out: list[FaceObs] = []
        if faces is None:
            return out
        for f in faces:
            fx, fy, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
            if fh / up < self.min_size:
                continue
            try:
                aligned = self.rec.alignCrop(sub, f)
                v = self._feature(aligned)
            except Exception:      # noqa: BLE001
                continue
            out.append(FaceObs(X1 + fx / up, Y1 + fy / up,
                               X1 + (fx + fw) / up, Y1 + (fy + fh) / up,
                               float(f[-1]), v))
        return out


def assign_to_tracks(faces: list[FaceObs], boxes: list[tuple[int, tuple]],
                     max_ratio: float = 0.42, center_tol: float = 0.22,
                     size_lo: float = 0.06, size_hi: float = 0.30) -> dict[int, FaceObs]:
    """把脸归属到人框，几何约束必须是严的。

    舞者密集交叉时，邻座的脸会落进本条轨迹的框里——这正是 44.5–56s 那次
    "人脸 0.84 却绑错人"的成因。三条硬约束：
      1. 脸中心必须在框的横向中间带（±22% 框宽）
      2. 脸中心必须在框顶往下 42% 以内
      3. 脸高 / 框高 比例合理（0.06–0.30）
    多个框同时命中时取面积最小的那个。
    """
    out: dict[int, FaceObs] = {}
    for f in faces:
        best, best_area = None, 1e18
        for key, b in boxes:
            x1, y1, x2, y2 = b
            bw, hh = x2 - x1, y2 - y1
            if bw < 8 or hh < 20:
                continue
            cx = (x1 + x2) / 2
            if abs(f.cx - cx) > bw * center_tol:
                continue
            if not (y1 - 0.02 * hh <= f.cy <= y1 + hh * max_ratio):
                continue
            ratio = f.h / hh
            if not (size_lo <= ratio <= size_hi):
                continue
            area = bw * hh
            if area < best_area:
                best, best_area = key, area
        if best is not None:
            out[best] = f
    return out
