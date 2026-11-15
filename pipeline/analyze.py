# -*- coding: utf-8 -*-
"""Pass 1 + Pass 2：视频分析。

产物（都落在 outputs/<stem>/ 下）：
  video_info.json   分辨率/帧率/帧数
  shots.json        镜头切分
  tracks.jsonl      逐帧局部轨迹（fi, t, 局部ID, bbox, 分数）
"""
from __future__ import annotations

import time
from pathlib import Path

from core.schema import read_json, write_json, write_jsonl
from core.video import frames, probe
from scene.shot_detector import detect as detect_shots


def run(cfg: dict, force: bool = False) -> Path:
    src = Path(cfg["input"]["video"]).resolve()
    out_dir = Path(cfg["output"]["dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe(src)
    write_json(out_dir / "video_info.json", dict(
        path=str(src), width=info.width, height=info.height, fps=info.fps,
        n_frames=info.n_frames, duration=info.duration,
        has_audio=info.has_audio, codec=info.codec))

    limit = int(cfg["input"].get("limit") or 0)
    t0 = time.time()

    # ---------------- Pass 1: 镜头
    p_shots = out_dir / "shots.json"
    if p_shots.exists() and not force:
        shots_raw = read_json(p_shots)
        print(f"[pass1] 复用 {p_shots.name}（{len(shots_raw)} 个镜头）")
    else:
        shots = detect_shots(src, info, limit=limit)
        shots_raw = [dict(shot_id=s.shot_id, start_frame=s.start_frame,
                          end_frame=s.end_frame, start_t=round(s.start_t, 3),
                          end_t=round(s.end_t, 3), confidence=round(s.confidence, 3))
                     for s in shots]
        write_json(p_shots, shots_raw)
        print(f"[pass1] 切分完成 {time.time()-t0:.1f}s，{len(shots_raw)} 个镜头")

    shot_of = {}
    for s in shots_raw:
        for fi in range(s["start_frame"], s["end_frame"]):
            shot_of[fi] = s["shot_id"]

    # ---------------- Pass 2: 检测 + 局部跟踪
    from detection.yolo_detector import UltralyticsTracker, YoloDetector

    dcfg, tcfg = cfg["detection"], cfg["tracking"]
    detector = YoloDetector(weights=dcfg["weights"], imgsz=int(dcfg["imgsz"]),
                            conf=float(dcfg["confidence"]), iou=float(dcfg.get("iou", 0.6)),
                            device=dcfg.get("device", 0), classes=tuple(dcfg.get("classes", [0])))
    tracker = UltralyticsTracker(detector, tcfg.get("tracker_cfg", "botsort.yaml"))

    rows = []
    cur_shot = None
    n_obs = 0
    t1 = time.time()
    total = min(limit, info.n_frames) if limit else info.n_frames

    for fi, frame in frames(src, limit=limit):
        sid = shot_of.get(fi, 0)
        if tcfg.get("reset_on_cut", True) and cur_shot is not None and sid != cur_shot:
            tracker.reset()
        cur_shot = sid
        t = fi / info.fps
        for o in tracker.step(frame, fi, t):
            o.shot_id = sid
            rows.append(o)
            n_obs += 1
        if fi % 100 == 0 and fi:
            el = time.time() - t1
            fps_now = fi / el if el else 0
            print(f"\r  {fi}/{total}  {fps_now:5.1f} FPS  目标 {n_obs}", end="", flush=True)

    el = time.time() - t1
    print(f"\r  {total}/{total}  {total/el:5.1f} FPS  轨迹点 {n_obs}" + " " * 20)

    p = out_dir / "tracks.jsonl"
    write_jsonl(p, (r.to_dict() for r in rows))
    write_json(out_dir / "analyze_stats.json", dict(
        frames=total, observations=n_obs,
        avg_per_frame=round(n_obs / max(1, total), 2),
        detect_fps=round(total / el, 2),
        detect_seconds=round(el, 1),
        shots=len(shots_raw),
        single_take=len(shots_raw) == 1,
        tracker=tracker.name, weights=str(dcfg["weights"]), imgsz=int(dcfg["imgsz"])))
    print(f"[pass2] {el:.1f}s  写入 {p.name}（{n_obs} 行）")
    return out_dir
