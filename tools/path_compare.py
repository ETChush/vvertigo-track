# -*- coding: utf-8 -*-
"""画出两种取景路径的对比：位置、速度、加速度，并标出关键帧位置。

用法：python tools/path_compare.py --outdir outputs/taq600
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, ".")

from camera import path as cam_path
from camera import planner
from core import config
from core.schema import IdentityObs, read_json, read_jsonl, write_json


def load_series(out_dir: Path, cfg: dict):
    info = read_json(out_dir / "video_info.json")
    rows = list(read_jsonl(out_dir / "target_tracks.jsonl"))
    n = int(info["n_frames"])
    fps = float(info["fps"])
    series: list[IdentityObs | None] = [None] * n
    for r in rows:
        series[int(r["fi"])] = IdentityObs(
            fi=int(r["fi"]), t=float(r["t"]),
            global_id=int(r.get("gid", r.get("global_id", 0))),
            local_id=int(r.get("lid", r.get("local_id", 0)) or 0),
            x1=r["x1"], y1=r["y1"], x2=r["x2"], y2=r["y2"],
            appearance_score=0.0, motion_score=0.0,
            identity_conf=float(r.get("conf", 1.0)),
            decision="BIND", shot_id=int(r.get("sid", 0)))
    return series, info, fps, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/taq600")
    ap.add_argument("--out", default="outputs/path_compare.png")
    ap.add_argument("--t0", type=float, default=40.0)
    ap.add_argument("--t1", type=float, default=70.0)
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    cfg = config.load("configs/default.yaml")
    series, info, fps, n = load_series(out_dir, cfg)
    ccfg = cfg["camera"]

    old_cfg = dict(ccfg, path_mode="one_euro")
    new_cfg = dict(ccfg, path_mode="keyframe")

    poses_old, sum_old = planner.plan(series, info, old_cfg, 0)
    poses_new, sum_new = planner.plan(series, info, new_cfg, 0)

    x_old = np.array([p.center_x for p in poses_old])
    x_new = np.array([p.center_x for p in poses_new])
    t = np.arange(n) / fps

    # 关键帧位置（重新算一遍，用于在图上标竖线）
    raw = np.array([s.cx if s is not None else np.nan for s in series])
    filled = np.where(np.isnan(raw), np.nan, raw)
    idx = np.flatnonzero(~np.isnan(filled))
    filled = np.interp(np.arange(n), idx, filled[idx]) if len(idx) else np.zeros(n)
    pre = cam_path.gaussian_filter1d(filled, ccfg.get("smooth_sigma", 16), mode="nearest")
    keys = cam_path.rdp(pre, ccfg.get("simplify_eps", 3.0), ccfg.get("time_weight", 16))

    a0, a1 = int(args.t0 * fps), int(args.t1 * fps)
    sl = slice(a0, a1)

    def vel(x):
        return np.gradient(x) * fps

    def acc(x):
        return np.gradient(np.gradient(x)) * fps * fps

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1.2, 1.2]))
    for ax in axes:
        ax.set_facecolor("#fafafa")
        ax.grid(alpha=0.25, lw=0.6)

    ax = axes[0]
    ax.plot(t[sl], x_old[sl], color="#d64545", lw=0.9, label="OLD  causal One-Euro")
    ax.plot(t[sl], x_new[sl], color="#2f7ed8", lw=1.6, label="NEW  keyframe + PCHIP spline")
    kt = t[keys[(keys >= a0) & (keys < a1)]]
    ax.vlines(kt, x_new[sl].min() - 10, x_new[sl].max() + 10,
              color="#2f7ed8", alpha=0.18, lw=0.8)
    ax.set_ylabel("camera center x (px)", fontsize=10)
    ax.set_title(f"Camera path  {args.t0:.0f}-{args.t1:.0f}s   "
                 f"keyframes: {sum_new['path']['keyframes']} total "
                 f"({sum_new['path']['frames_per_keyframe']} frames each)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax = axes[1]
    ax.plot(t[sl], vel(x_old)[sl], color="#d64545", lw=0.7, alpha=0.85, label="OLD")
    ax.plot(t[sl], vel(x_new)[sl], color="#2f7ed8", lw=1.3, label="NEW")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_ylabel("velocity (px/s)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax = axes[2]
    ax.plot(t[sl], acc(x_old)[sl], color="#d64545", lw=0.6, alpha=0.7, label="OLD")
    ax.plot(t[sl], acc(x_new)[sl], color="#2f7ed8", lw=1.1, label="NEW")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_ylabel("accel (px/s²)", fontsize=10)
    ax.set_xlabel("time (s)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(-4200, 4200)

    txt = (f"OLD: reversals {sum_old['reversals']}, |a|p95 {np.percentile(np.abs(acc(x_old)),95):.0f} px/s², "
           f"mean |v| {sum_old['mean_speed_px_s']:.0f} px/s\n"
           f"NEW: reversals {sum_new['reversals']}, |a|p95 {sum_new['path']['accel_p95']:.0f} px/s², "
           f"mean |v| {sum_new['mean_speed_px_s']:.0f} px/s, jerk p50 {sum_new['path']['jerk_p50']:.0f}")
    fig.text(0.012, 0.005, txt, fontsize=8.5, color="#333", family="monospace")

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, facecolor="white")
    print(f"saved {out}")
    print(f"OLD  reversals={sum_old['reversals']}  mean|v|={sum_old['mean_speed_px_s']}  "
          f"|a|p95={np.percentile(np.abs(acc(x_old)),95):.0f}")
    print(f"NEW  reversals={sum_new['reversals']}  mean|v|={sum_new['mean_speed_px_s']}  "
          f"|a|p95={sum_new['path']['accel_p95']}  jerk_p50={sum_new['path']['jerk_p50']}  "
          f"kf={sum_new['path']['keyframes']}")


if __name__ == "__main__":
    main()
