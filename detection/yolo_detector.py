# -*- coding: utf-8 -*-
"""YOLO 检测后端。可直接用（detect），也可被跟踪器复用（持有同一个模型实例）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.schema import TrackObs


class YoloDetector:
    name = "yolo"

    def __init__(self, weights: str, imgsz: int = 1280, conf: float = 0.25,
                 iou: float = 0.6, device=0, classes: tuple[int, ...] = (0,),
                 half: bool = True):
        from ultralytics import YOLO  # 延迟导入，方便无 GPU 环境跑其它模块

        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.classes = list(classes)
        self.half = half
        self._warm = False

    def _kw(self) -> dict:
        return dict(imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                    classes=self.classes, device=self.device, verbose=False)

    def detect(self, frame: np.ndarray, fi: int = 0, t: float = 0.0) -> list[TrackObs]:
        r = self.model.predict(frame, **self._kw())[0]
        return self._to_obs(r, fi, t)

    def _to_obs(self, r, fi: int, t: float, with_id: bool = False) -> list[TrackObs]:
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        ids = (r.boxes.id.cpu().numpy().astype(int)
               if with_id and r.boxes.id is not None else None)
        out = []
        for i, (b, s) in enumerate(zip(xyxy, conf)):
            out.append(TrackObs(fi=fi, t=t,
                                local_id=int(ids[i]) if ids is not None else -1,
                                x1=float(b[0]), y1=float(b[1]),
                                x2=float(b[2]), y2=float(b[3]),
                                score=float(s)))
        return out


class UltralyticsTracker:
    """ultralytics 内置跟踪器（bytetrack / botsort）。

    注意：with_reid=True 时它用的是检测器自带特征，不是独立 ReID 模型，
    效果弱于专门的检索式 ReID；本项目把跨镜头身份交给 identity/ 处理。
    """

    def __init__(self, detector: YoloDetector, tracker_cfg: str = "botsort.yaml"):
        self.detector = detector
        self.tracker_cfg = tracker_cfg
        self.name = tracker_cfg.replace(".yaml", "")

    def step(self, frame: np.ndarray, fi: int, t: float) -> list[TrackObs]:
        r = self.detector.model.track(frame, persist=True,
                                      tracker=self.tracker_cfg,
                                      **self.detector._kw())[0]
        return self.detector._to_obs(r, fi, t, with_id=True)

    def reset(self) -> None:
        pred = getattr(self.detector.model, "predictor", None)
        for tr in getattr(pred, "trackers", []) or []:
            tr.reset()
