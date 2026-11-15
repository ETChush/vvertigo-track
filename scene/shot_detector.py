# -*- coding: utf-8 -*-
"""镜头切分。

不装 PySceneDetect（它的 ContentDetector 本质上也是这个思路）。舞蹈片有两个
难点：快速肢体动作会制造大像素差，而真实切镜的直方图几乎必然突变。所以用
「直方图距离为主 + 像素差为辅」，并把低置信的候选标成 suspicious 留待人工确认。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.schema import Shot, VideoInfo

W_SMALL, H_SMALL = 96, 54


def _feature(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(frame, (W_SMALL, H_SMALL), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return hist.flatten(), gray.astype(np.float32)


def detect(video: str | Path, info: VideoInfo, sample_every: int = 1,
           hist_thresh: float = 0.42, pix_thresh: float = 22.0,
           min_gap_s: float = 0.35, limit: int = 0) -> list[Shot]:
    """返回按时间排序的镜头列表，覆盖 [0, n_frames)。"""
    cap = cv2.VideoCapture(str(video))
    fps = info.fps or 30.0
    total = info.n_frames or 0
    min_gap = max(1, int(round(min_gap_s * fps)))

    prev_hist = prev_gray = None
    cuts: list[tuple[int, float]] = []
    fi = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if limit and fi >= limit:
            break
        if fi % sample_every:
            continue
        hist, gray = _feature(frame)
        if prev_gray is not None:
            hd = float(cv2.compareHist(prev_hist.reshape(24, 16), hist.reshape(24, 16),
                                       cv2.HISTCMP_BHATTACHARYYA))
            pd = float(np.mean(np.abs(gray - prev_gray)))
            strong = hd >= hist_thresh and pd >= pix_thresh
            weak = hd >= hist_thresh * 1.35
            if strong or weak:
                conf = min(1.0, 0.65 * min(1.0, hd / 0.6) + 0.35 * min(1.0, pd / 40.0))
                if cuts and fi - cuts[-1][0] < min_gap:
                    if conf > cuts[-1][1]:
                        cuts[-1] = (fi, conf)
                else:
                    cuts.append((fi, conf))
        prev_hist, prev_gray = hist, gray
    cap.release()

    if limit:
        total = fi + 1
    n = total or (fi + 1)
    bounds = [0] + [c[0] for c in cuts if 0 < c[0] < n] + [n]
    shots: list[Shot] = []
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        if b <= a:
            continue
        conf = 1.0 if k == 0 else next((c[1] for c in cuts if c[0] == a), 1.0)
        shots.append(Shot(shot_id=len(shots), start_frame=a, end_frame=b,
                          start_t=a / fps, end_t=b / fps, confidence=conf))
    return shots


def shot_at(shots: list[Shot], fi: int) -> Shot:
    for s in shots:
        if s.contains(fi):
            return s
    return shots[-1]
