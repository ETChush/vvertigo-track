# -*- coding: utf-8 -*-
"""把两次运行的质量报告拉平成一张表，做同尺子对比。

  python tools/compare_runs.py outputs/seg_002 outputs/taq600
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import read_json    # noqa: E402

ROWS = [
    ("跟踪器/来源", lambda r: (r.get("analyze_stats", {}).get("tracker")
                           or r.get("analyze_stats", {}).get("source") or "?")),
    ("局部轨迹数", lambda r: r.get("identity_stats", {}).get("n_local_tracks")),
    ("全局身份数", lambda r: r.get("identity_stats", {}).get("n_identities")),
    ("有脸轨迹数", lambda r: r.get("identity_stats", {}).get("tracks_with_face")),
    ("目标可见率", lambda r: _pct(r.get("identity_stats", {}).get("target", {}).get("coverage"))),
    ("最长空洞(s)", lambda r: r.get("identity_stats", {}).get("target", {}).get("longest_gap_s")),
    ("目标局部轨迹数", lambda r: r.get("identity_stats", {}).get("target", {}).get("local_tracks")),
    ("可疑换人", lambda r: r.get("identity_stats", {}).get("swap_suspects")),
    ("全身保留", lambda r: _pct(r.get("camera", {}).get("target_inside_ratio"))),
    ("镜头反向次数", lambda r: r.get("camera", {}).get("reversals")),
    ("均速(px/s)", lambda r: r.get("camera", {}).get("mean_speed_px_s")),
    ("裁切+放大", lambda r: (f"{r.get('camera',{}).get('crop_w')}x{r.get('camera',{}).get('crop_h')}"
                          f" @{r.get('camera',{}).get('upscale')}x")),
    ("渲染帧率", lambda r: r.get("render", {}).get("fps")),
    ("成片大小(MB)", lambda r: r.get("render", {}).get("size_mb")),
]


def _pct(v):
    return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()

    runs = []
    for d in args.dirs:
        p = Path(d) / "quality_report.json"
        if not p.exists():
            print(f"[skip] {d} 没有 quality_report.json")
            continue
        runs.append((Path(d).name, read_json(p)))

    if len(runs) < 2:
        print("至少需要两份报告")
        return

    width = 22
    head = f"{'指标':<12}" + "".join(f"{n:>{width}}" for n, _ in runs)
    print(head)
    print("-" * len(head))
    for name, fn in ROWS:
        vals = []
        for _, r in runs:
            try:
                v = fn(r)
            except Exception:      # noqa: BLE001
                v = "-"
            vals.append("-" if v is None else str(v))
        print(f"{name:<12}" + "".join(f"{v:>{width}}" for v in vals))


if __name__ == "__main__":
    main()
