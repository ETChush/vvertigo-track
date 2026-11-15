# -*- coding: utf-8 -*-
"""YuNet + SFace 的 onnxruntime 版本（优先 CUDA）。

为什么值得单独写一遍
------------------
`cv2.FaceDetectorYN` 是纯 CPU 的。实测一帧 10 个头部裁剪里，YuNet 检测占了
**77.9%**（33.7ms/框），SFace 只占 19%——所以换 GPU 必须换 YuNet，光把识别
挪到 GPU 没有意义。

同一个 yunet.onnx 在 CUDA 上单次前向 **3.0ms**（640x640），约 11 倍。

两个前提必须守住，否则所有历史阈值全部失效：
  1. 用**同一份** yunet.onnx / sface.onnx，不换模型；
  2. 预处理要和 OpenCV 内部一致（YuNet: BGR 0-255 直接拉伸到 640；
     SFace: (x-127.5)/128 且 BGR→RGB），否则特征空间会漂。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_ORT = None


def _load_cudnn() -> None:
    """把 torch 自带的 cuDNN 目录挂进 DLL 搜索路径。

    onnxruntime-gpu 需要 cuDNN 9，但系统里通常没有；torch 的 cu124 wheel 自带
    一份，把它挂上去就不用单独装 CUDA Toolkit。挂不上也没关系，会退回 CPU。
    """
    try:
        import torch

        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir():
            try:
                os.add_dll_directory(str(lib))
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception:                                   # noqa: BLE001
        pass


def get_ort():
    """返回 onnxruntime 模块，优先 CUDA；没有 GPU 版就返回 None。"""
    global _ORT
    if _ORT is not None:
        return _ORT or None
    _load_cudnn()
    try:
        import onnxruntime as ort
    except ImportError:
        _ORT = False
        return None
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        _ORT = False
        return None
    _ORT = ort
    return ort


class OnnxYuNet:
    """YuNet 640x640 推理与解码。

    解码公式来自 OpenCV 的 YuNet 实现（face_detect.cpp）：网格中心 + 对数宽高。
    """

    STRIDES = (8, 16, 32)

    def __init__(self, path: str, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        ort = get_ort()
        if ort is None:
            raise RuntimeError("onnxruntime 没有 CUDA provider")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=list(providers))
        self.in_name = self.sess.get_inputs()[0].name
        self.size = 640

    def detect(self, img: np.ndarray, score_thresh: float = 0.6,
               nms_thresh: float = 0.3) -> np.ndarray:
        """返回 (N, 15)：x, y, w, h, 5 个关键点(10), score。坐标是 img 的像素。

        缩放用**保持比例的 letterbox**，不是直接拉伸。4K 拉伸到 640x640 会把
        人脸压扁 1.78 倍，实测框的 IoU 只有 0.55；letterbox 是 0.88。
        """
        import cv2

        ih, iw = img.shape[:2]
        scale = min(self.size / iw, self.size / ih)
        nw, nh = max(2, int(round(iw * scale))), max(2, int(round(ih * scale)))
        padx, pady = (self.size - nw) // 2, (self.size - nh) // 2
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        canvas[pady:pady + nh, padx:padx + nw] = cv2.resize(
            img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = canvas.astype(np.float32).transpose(2, 0, 1)[None]
        outs = self.sess.run(None, {self.in_name: blob})
        d = dict(zip([o.name for o in self.sess.get_outputs()], outs))

        boxes, scores, kpss = [], [], []
        for s in self.STRIDES:
            cls = d[f"cls_{s}"][0, :, 0]
            obj = d[f"obj_{s}"][0, :, 0]
            bbox = d[f"bbox_{s}"][0]
            kps = d[f"kps_{s}"][0]
            sc = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            idx = np.nonzero(sc >= score_thresh)[0]
            if len(idx) == 0:
                continue
            cols = self.size // s
            gy, gx = np.divmod(idx, cols)
            b = bbox[idx]
            k = kps[idx]
            px = (gx + b[:, 0]) * s - padx
            py = (gy + b[:, 1]) * s - pady
            pw = np.exp(b[:, 2]) * s
            ph = np.exp(b[:, 3]) * s
            boxes.append(np.stack([px - pw / 2, py - ph / 2, pw, ph], 1))
            scores.append(sc[idx])
            kp = np.empty((len(idx), 10), np.float32)
            for j in range(5):
                kp[:, 2 * j] = (gx + k[:, 2 * j]) * s - padx
                kp[:, 2 * j + 1] = (gy + k[:, 2 * j + 1]) * s - pady
            kpss.append(kp)

        if not boxes:
            return np.zeros((0, 15), np.float32)
        boxes = np.concatenate(boxes)
        scores = np.concatenate(scores)
        kpss = np.concatenate(kpss)

        rects = [[float(x), float(y), float(w), float(h)] for x, y, w, h in boxes]
        keep = cv2.dnn.NMSBoxes(rects, scores.tolist(), score_thresh, nms_thresh)
        if len(keep) == 0:
            return np.zeros((0, 15), np.float32)
        keep = np.array(keep).reshape(-1)

        inv = 1.0 / scale
        out = np.concatenate([boxes[keep], kpss[keep], scores[keep, None]], 1).copy()
        out[:, [0, 1, 2, 3]] *= inv
        for j in range(5):
            out[:, 4 + 2 * j] *= inv
            out[:, 5 + 2 * j] *= inv
        return out.astype(np.float32)


class OnnxSFace:
    """SFace 特征提取。预处理与 OpenCV 的 FaceRecognizerSF 保持一致。"""

    def __init__(self, path: str, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        ort = get_ort()
        if ort is None:
            raise RuntimeError("onnxruntime 没有 CUDA provider")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=list(providers))
        self.in_name = self.sess.get_inputs()[0].name

    def feature(self, aligned: np.ndarray) -> np.ndarray:
        import cv2

        if aligned.shape[:2] != (112, 112):
            aligned = cv2.resize(aligned, (112, 112), interpolation=cv2.INTER_LINEAR)
        # 实测：这个 sface.onnx 吃的是**未归一化**的 RGB 0-255。
        # 逐个试过 (x±127.5)/128 和 /255，余弦只有 0.0~0.1；raw RGB 是 1.00000。
        # 归一化层在模型内部，外部再除一次就把特征空间毁了。
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = rgb.transpose(2, 0, 1)[None]
        v = self.sess.run(None, {self.in_name: blob})[0].flatten().astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


def align_face(img: np.ndarray, kps: np.ndarray, out: int = 112) -> np.ndarray:
    """按 5 点做相似变换对齐，与 OpenCV FaceRecognizerSF::alignCrop 同款。

    目标点位用 ArcFace 的 112x112 标准模板（SFace 训练时就是这一套）。
    """
    import cv2

    std = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
    src = np.asarray(kps, np.float32).reshape(5, 2)
    M, _ = cv2.estimateAffinePartial2D(src, std, method=cv2.LMEDS)
    if M is None:
        return cv2.resize(img, (out, out))
    return cv2.warpAffine(img, M, (out, out), borderValue=0.0)
