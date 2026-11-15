# -*- coding: utf-8 -*-
"""Pass 4：摄影机规划。只读目标轨迹，产出 camera_path.json。"""
from __future__ import annotations

from pathlib import Path

from camera import planner
from core.schema import IdentityObs, read_json, read_jsonl, write_json


def load_series(path: Path, n_frames: int) -> list[IdentityObs | None]:
    series: list[IdentityObs | None] = [None] * n_frames
    for r in read_jsonl(path):
        fi = int(r["fi"])
        if 0 <= fi < n_frames:
            series[fi] = IdentityObs(fi=fi, t=float(r["t"]), global_id=int(r["gid"]),
                                     local_id=int(r["lid"]), x1=float(r["x1"]),
                                     y1=float(r["y1"]), x2=float(r["x2"]),
                                     y2=float(r["y2"]), identity_conf=float(r["conf"]),
                                     shot_id=int(r.get("sid", 0)))
    return series


def run(cfg: dict) -> Path:
    out_dir = Path(cfg["output"]["dir"]).resolve()
    info = read_json(out_dir / "video_info.json")
    n_frames = int(info["n_frames"])
    if cfg["input"].get("limit"):
        n_frames = min(n_frames, int(cfg["input"]["limit"]))
    series = load_series(out_dir / "target_tracks.jsonl", n_frames)

    istats = read_json(out_dir / "identity_stats.json") if (out_dir / "identity_stats.json").exists() else {}
    target_id = int(istats.get("target", {}).get("global_id", -1))

    ccfg = dict(cfg["camera"])
    ccfg["out_width"] = int(cfg["export"]["width"])
    poses, summary = planner.plan(series, info, ccfg, target_id)

    write_json(out_dir / "camera_path.json", [p.to_dict() for p in poses])
    write_json(out_dir / "camera_stats.json", summary)
    print(f"[pass4] {len(poses)} 帧取景  裁切 {summary['crop_w']}x{summary['crop_h']}  "
          f"放大 {summary['upscale']}x  状态 {summary['states']}")
    print(f"[pass4] 目标可见 {summary['visible_ratio']*100:.1f}%  "
          f"裁切内含全身 {summary['target_inside_ratio']*100:.1f}%  "
          f"反向 {summary['reversals']} 次")
    return out_dir
