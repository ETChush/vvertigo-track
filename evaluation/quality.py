# -*- coding: utf-8 -*-
"""质量检测与报告。

不需要标注真值就能算的指标（客观、可复现）：
  * 目标可见率、最大空洞
  * 全身保留（裁切框是否完整包含人体框的水平范围；纵向满幅，天然满足）
  * 摄影机运动学（速度、反向次数 = 抖动代理指标）
  * 跟错人的代理指标：身份事件里的可疑换人数

真正的 FBR / IDF1 需要人工真值，那是 evaluation/annotation.py 的事，这里不做假数据。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.schema import CameraPose, read_jsonl, write_json


def _thumbs(frames: list[np.ndarray], cols: int, tile_w: int) -> np.ndarray:
    h, w = frames[0].shape[:2]
    tile_h = int(tile_w * h / w)
    rows = (len(frames) + cols - 1) // cols
    sheet = np.zeros((tile_h * rows, tile_w * cols, 3), np.uint8)
    for k, f in enumerate(frames):
        r, c = divmod(k, cols)
        sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = cv2.resize(
            f, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    return sheet


def contact_sheet(src: str, poses: list[CameraPose], out: str | Path,
                  picks: int = 12) -> Path:
    cap = cv2.VideoCapture(src)
    n = len(poses)
    idx = [int(i * (n - 1) / max(1, picks - 1)) for i in range(picks)]
    tiles = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        p = poses[i]
        crop = frame[p.crop_y:p.crop_y + p.crop_h, p.crop_x:p.crop_x + p.crop_w]
        crop = cv2.resize(crop, (270, 480), interpolation=cv2.INTER_AREA)
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (60, 60, 60), 1)
        label = f"{p.t:5.1f}s {p.state[:4]}"
        cv2.rectangle(crop, (0, 0), (150, 22), (0, 0, 0), -1)
        cv2.putText(crop, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)
        tiles.append(crop)
    cap.release()
    if not tiles:
        raise SystemExit("contact sheet: 没有取到帧")
    sheet = _thumbs(tiles, cols=4, tile_w=270)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return p


def framing_check(src: str, poses: list[CameraPose], series, out: str | Path,
                  picks: int = 6) -> Path:
    """整帧 + 裁切框 + 目标框，用来人眼确认"有没有切到人"。"""
    cap = cv2.VideoCapture(src)
    n = len(poses)
    idx = [int(i * (n - 1) / max(1, picks - 1)) for i in range(picks)]
    tiles = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        p = poses[i]
        vis = frame.copy()
        cv2.rectangle(vis, (p.crop_x, p.crop_y),
                      (p.crop_x + p.crop_w, p.crop_y + p.crop_h), (60, 220, 255), 4)
        s = series[i] if i < len(series) else None
        if s is not None:
            cv2.rectangle(vis, (int(s.x1), int(s.y1)), (int(s.x2), int(s.y2)),
                          (60, 255, 120), 3)
        cv2.putText(vis, f"{p.t:.1f}s {p.state}", (p.crop_x + 10, p.crop_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 220, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(vis, (640, 360), interpolation=cv2.INTER_AREA))
    cap.release()
    sheet = _thumbs(tiles, cols=2, tile_w=640)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return p


def report(out_dir: Path, extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    stats = {}
    for name in ("analyze_stats", "identity_stats"):
        p = out_dir / f"{name}.json"
        if p.exists():
            from core.schema import read_json
            stats[name] = read_json(p)
    cam_p = out_dir / "camera_stats.json"
    if cam_p.exists():
        from core.schema import read_json
        stats["camera"] = read_json(cam_p)
    if extra:
        stats["render"] = extra

    ident = stats.get("identity_stats", {})
    cam = stats.get("camera", {})
    # 选人/清洗阶段会更新 target_summary.json，报告以它为准
    if (out_dir / "target_summary.json").exists():
        from core.schema import read_json as _rj
        stats["target"] = _rj(out_dir / "target_summary.json")
    tgt = stats.get("target") or ident.get("target", {})
    verdict = []
    if tgt:
        cov = tgt.get("coverage", 0) * 100
        verdict.append(f"目标可见 {cov:.1f}%（最长空洞 {tgt.get('longest_gap_s', 0)}s）")
    if cam:
        verdict.append(f"全身保留 {cam.get('target_inside_ratio', 0)*100:.1f}%"
                       f"（{cam.get('crop_w')}×{cam.get('crop_h')} 裁切，放大 {cam.get('upscale')}x）")
        verdict.append(f"摄影机反向 {cam.get('reversals')} 次，均速 "
                       f"{cam.get('mean_speed_px_s')} px/s")
    if ident:
        verdict.append(f"可疑换人 {ident.get('swap_suspects', 0)} 次")

    stats["summary_lines"] = verdict
    write_json(out_dir / "quality_report.json", stats)
    return stats
