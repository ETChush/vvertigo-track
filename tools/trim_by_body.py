# -*- coding: utf-8 -*-
"""按人体 ReID 给目标序列做逐帧清洗（发布前的质量闸门）。

与 identity/refine.py 的区别：refine 只能切"有关联样本"的位置；当人和别人贴得太近、
关联被丢弃时就没有样本可判，尾巴会漏过去。这个 pass 直接对每一帧取最佳匹配的人体
特征打分，把明显不是本人的帧剪掉——剪掉的帧在成片里表现为"停住"，不会跟错人。

  python tools/trim_by_body.py --outdir outputs/seg_002
  python main.py --stage plan,render
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import read_json, read_jsonl, write_json, write_jsonl   # noqa: E402
from core.video import frames as iter_frames                              # noqa: E402
from identity.person_analysis import PersonAnalyzer, iou                  # noqa: E402


def norm(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/seg_002")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--thresh", type=float, default=0.62)
    ap.add_argument("--min-iou", type=float, default=0.45)
    ap.add_argument("--ref-frames", type=int, default=120,
                    help="用目标最早出现的前 N 个可见帧建立基准")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = Path(args.outdir)
    info = read_json(d / "video_info.json")
    fps = float(info["fps"])
    rows = list(read_jsonl(d / "target_tracks.jsonl"))
    if not rows:
        raise SystemExit("没有 target_tracks.jsonl")
    by_fi = {r["fi"]: r for r in rows}
    order = sorted(by_fi)[: args.ref_frames]

    app = PersonAnalyzer()

    # 1) 基准：目标最早的一批帧
    refs = []
    for fi, frame in iter_frames(info["path"], limit=order[-1] + 1):
        if fi not in order:
            continue
        r = by_fi[fi]
        tb = (r["x1"], r["y1"], r["x2"], r["y2"])
        for p in app.get(frame):
            if p.reid is not None and iou(p.bbox, tb) > args.min_iou:
                refs.append(norm(p.reid))
                break
    if not refs:
        raise SystemExit("基准帧没取到人体特征")
    ref = norm(np.mean(np.stack(refs), axis=0))
    print(f"基准: 前 {len(refs)} 帧的人体特征")

    # 2) 逐帧打分
    bad = set()
    scored = 0
    for fi, frame in iter_frames(info["path"], limit=int(info["n_frames"])):
        if fi % args.every:
            continue
        r = by_fi.get(fi)
        if r is None:
            continue
        tb = (r["x1"], r["y1"], r["x2"], r["y2"])
        best = None
        for p in app.get(frame):
            if p.reid is None:
                continue
            v = iou(p.bbox, tb)
            if v > args.min_iou and (best is None or v > best[0]):
                best = (v, p)
        if best is None:
            continue
        scored += 1
        if float(ref @ norm(best[1].reid)) < args.thresh:
            bad.add(fi)
    app.close()

    if not bad:
        print(f"打分 {scored} 帧，没有需要剪掉的帧")
        return

    # 3) 合并成区间并剪掉
    flags = sorted(bad)
    spans, cur = [], [flags[0], flags[0]]
    for fi in flags[1:]:
        if fi - cur[1] <= args.every * 2:
            cur[1] = fi
        else:
            spans.append(tuple(cur))
            cur = [fi, fi]
    spans.append(tuple(cur))
    dropped = [r for r in rows if any(a <= r["fi"] <= b for a, b in spans)]
    print(f"打分 {scored} 帧，异常 {len(bad)} 帧，合并为 {len(spans)} 段")
    for a, b in spans:
        print(f"  剪掉 {a/fps:6.1f}s - {b/fps:6.1f}s  ({(b-a)/fps:.1f}s)")

    if args.dry_run:
        return
    keep = [r for r in rows if not any(a <= r["fi"] <= b for a, b in spans)]
    write_jsonl(d / "target_tracks.jsonl", keep)
    rep = d / "trim_report.json"
    write_json(rep, dict(scored=scored, bad=len(bad), dropped=dropped.__len__(),
                         kept=len(keep), spans=[[a, b] for a, b in spans],
                         thresh=args.thresh))
    print(f"保留 {len(keep)} / {len(rows)} 帧，写回 target_tracks.jsonl")


if __name__ == "__main__":
    main()
