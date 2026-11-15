# -*- coding: utf-8 -*-
"""快速检测：只要框，不要轨迹。

anchor 模式（人脸锚点 + 局部框关联）根本不需要跟踪器的 ID 连续性——它只要求
每一帧有足够准的人体框。所以整条 BoT-SORT/TAQ 链路都可以跳过，直接：

    ffmpeg 管道（缩到 1280 宽）→ YOLO 批量推理 → 坐标还原 → tracks.jsonl

实测收益：OpenCV 解码 22.6 FPS → 管道 162 FPS，YOLO 1280 本身 65 FPS，
于是检测这一段从 4~5 FPS 提到 60 FPS 以上。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from core.fastvideo import pipe_frames, probe_size
from core.schema import write_json, write_jsonl, read_jsonl
from core.video import probe


def run(video: str | Path, out_dir: str | Path, *, det_w: int = 1280,
        batch: int = 8, conf: float = 0.25, iou: float = 0.6,
        weights: str = "", device: int = 0, half: bool = True,
        limit: int = 0, log=print) -> dict:
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    try:
        from ultralytics.utils import LOGGER
        LOGGER.setLevel(logging.ERROR)
    except Exception:                                  # noqa: BLE001
        pass
    from ultralytics import YOLO

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ow, oh, fps = probe_size(video)
    if not det_w or det_w >= ow:
        det_w = ow                      # 0 = 和源一样，交给 YOLO 自己 letterbox
    det_h = int(round(det_w * oh / ow / 2)) * 2
    sx, sy = ow / det_w, oh / det_h
    log(f"  检测流 {det_w}x{det_h}（源 {ow}x{oh}，还原系数 {sx:.3f}），batch={batch}")

    model = YOLO(weights or "yolo11m.pt")
    kw = dict(imgsz=det_w, conf=conf, iou=iou, classes=[0], device=device,
              verbose=False, half=half)

    # —— 断点续跑：上一轮跑到一半被中断，.partial 里已有结果就接着跑。
    # 检测是流水线里最可能被中断的一段（几百帧才 flush 一次，中途关窗整段白跑），
    # 边跑边把已产生的框追加落盘，下次从断点继续。
    partial = out / "tracks.jsonl.partial"
    rows: list[dict] = []
    resume_fi = 0                       # 已经检测过的帧数（不含最后未 flush 的一批）
    if partial.exists():
        try:
            rows = read_jsonl(partial)
            if rows:
                resume_fi = max(int(r["fi"]) for r in rows) + 1
                log(f"  检测断点续跑：已检测到第 {resume_fi} 帧，从这继续")
        except (ValueError, OSError):
            rows = []
            resume_fi = 0

    buf: list[np.ndarray] = []
    t0 = time.time()
    n_frames = 0
    # 断点续跑时跳过前面已经检测过的帧，只读剩余部分
    start_fi = resume_fi
    for fi, frame in pipe_frames(video, width=det_w, height=det_h, count=limit):
        if fi < start_fi:
            continue                    # 已经检测过，跳过
        buf.append(frame.copy())
        n_frames = fi + 1
        if len(buf) < batch:
            continue
        _flush(model, kw, buf, rows, n_frames - len(buf), fps, sx, sy)
        buf.clear()
        if n_frames % 480 == 0:
            el = time.time() - t0
            # 每 480 帧把结果落盘一次，中断也能续跑
            write_jsonl(partial, rows)
            remain = (limit - n_frames) if limit else 0
            eta = "" if not limit else f"  剩 {remain / max(n_frames / el, 1e-6) / 60:.1f} 分钟"
            log(f"  {n_frames} 帧  {n_frames / el:5.1f} FPS  框 {len(rows)}{eta}")
    if buf:
        _flush(model, kw, buf, rows, n_frames - len(buf), fps, sx, sy)

    el = time.time() - t0
    write_jsonl(out / "tracks.jsonl", rows)
    if partial.exists():
        partial.unlink()                # 完成，清掉断点文件
    vi = probe(video)
    write_json(out / "video_info.json", dict(
        path=str(Path(video).resolve()), width=vi.width, height=vi.height,
        fps=vi.fps, n_frames=n_frames, duration=round(n_frames / fps, 3),
        has_audio=vi.has_audio, codec=vi.codec))
    write_json(out / "shots.json", [dict(shot_id=0, start_frame=0, end_frame=n_frames,
                                         start_t=0.0, end_t=round(n_frames / fps, 3),
                                         confidence=1.0)])
    stats = dict(frames=n_frames, observations=len(rows),
                 avg_per_frame=round(len(rows) / max(1, n_frames), 2),
                 detect_fps=round(n_frames / el, 2), detect_seconds=round(el, 1),
                 det_size=f"{det_w}x{det_h}", batch=batch, source="fast_detect")
    write_json(out / "analyze_stats.json", stats)
    log(f"  检测完成 {n_frames} 帧 {el:.1f}s = {n_frames / el:.1f} FPS，框 {len(rows)} 个")
    return stats


def _flush(model, kw, frames: list[np.ndarray], rows: list[dict], fi0: int,
           fps: float, sx: float, sy: float) -> None:
    res = model.predict(frames, **kw)
    for k, r in enumerate(res):
        fi = fi0 + k
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        for j, (b, s) in enumerate(zip(xyxy, cf)):
            rows.append(dict(fi=fi, t=round(fi / fps, 4), lid=j, sid=0,
                             x1=round(float(b[0]) * sx, 1), y1=round(float(b[1]) * sy, 1),
                             x2=round(float(b[2]) * sx, 1), y2=round(float(b[3]) * sy, 1),
                             s=round(float(s), 3)))
