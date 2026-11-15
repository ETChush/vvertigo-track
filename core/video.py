# -*- coding: utf-8 -*-
"""视频读取与元信息。统一用 ffprobe 取真实帧率/帧数，避免 cv2 猜错。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from core.schema import VideoInfo


def probe(path: str | Path) -> VideoInfo:
    p = Path(path)
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format", str(p)]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                         errors="ignore").stdout
    data = json.loads(out or "{}")
    vstream = next((s for s in data.get("streams", [])
                    if s.get("codec_type") == "video"), {})
    astream = next((s for s in data.get("streams", [])
                    if s.get("codec_type") == "audio"), None)

    num, den = (vstream.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den or 1)
    if fps <= 0 or fps > 500:
        fps = 30.0

    n_frames = int(vstream.get("nb_frames") or 0)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    if n_frames <= 0 and duration > 0:
        n_frames = int(round(duration * fps))
    if duration <= 0 and n_frames > 0:
        duration = n_frames / fps

    return VideoInfo(path=str(p.resolve()),
                     width=int(vstream.get("width") or 0),
                     height=int(vstream.get("height") or 0),
                     fps=fps, n_frames=n_frames, duration=duration,
                     has_audio=astream is not None,
                     codec=vstream.get("codec_name") or "")


def open_capture(path: str | Path) -> "cv2.VideoCapture":
    """打开解码器，优先走硬件加速。

    4K HEVC 60fps 上，默认解码只有 19 FPS，开 hwaccel 后 54 FPS（实测同一台机器）。
    整条链路的耗时大头都在解码上，所以统一从这里开。
    """
    p = str(path)
    try:
        cap = cv2.VideoCapture(p, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_HW_ACCELERATION,
                                cv2.VIDEO_ACCELERATION_ANY])
        if cap.isOpened():
            # 打开成功 ≠ 能解码。AV1 10bit 上 hwaccel 会 open 成功、read 直接返回
            # 空帧（ffmpeg 报 "HW accel end frame fail"），结果是整片检测出 0 个框
            # 还白跑一小时。这里试读一帧再决定。
            ok, f = cap.read()
            if ok and f is not None and getattr(f, "size", 0):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return cap
        cap.release()
    except Exception:      # noqa: BLE001
        pass
    return cv2.VideoCapture(p)


def frames(path: str | Path, limit: int = 0, skip: int = 0) -> Iterator[tuple[int, np.ndarray]]:
    """逐帧产出 (index, BGR)。index 从 0 开始。"""
    cap = open_capture(path)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {path}")
    if skip > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
    i = skip
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
            if limit and i - skip >= limit:
                break
    finally:
        cap.release()


def read_frame_at(path: str | Path, index: int) -> np.ndarray | None:
    cap = open_capture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None
