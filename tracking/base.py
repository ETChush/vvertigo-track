# -*- coding: utf-8 -*-
"""跟踪器抽象层。

所有后端必须实现同样的两步接口：step() 吐出一帧的 TrackObs，reset() 清空状态。
加一个新跟踪器（ByteTrack / BoT-SORT / OC-SORT / TAQ-MOTR 适配器）只需要实现这
两个方法，虚拟摄影机和渲染层完全不需要知道背后是什么。
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from core.schema import TrackObs


class TrackerBackend(Protocol):
    name: str

    def step(self, frame: np.ndarray, fi: int, t: float) -> list[TrackObs]:
        ...

    def reset(self) -> None:
        ...


class DetectionOnly:
    """只检测不关联的退化后端，用于基线对照（local_id 恒等于 -1）。"""

    name = "detect-only"

    def __init__(self, detector):
        self.detector = detector

    def step(self, frame, fi, t) -> list[TrackObs]:
        obs = self.detector.detect(frame, fi, t)
        for o in obs:
            o.local_id = fi * 1000 + int(o.cx)   # 造一个不会跨帧延续的 id
        return obs

    def reset(self) -> None:
        pass
