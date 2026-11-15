# -*- coding: utf-8 -*-
"""用人体 ReID 量化"跟错人"的时长占比。

为什么不用人脸：46px 的人脸单帧相似度噪声太大，会把"同一个人转头"算成"换人"。
人体 ReID 在同一人 0.8 / 不同人 0.5 附近的分离度更稳定，适合做这个判据。

  python tools/wrong_person_rate.py --outdir outputs/seg_002
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import read_json, read_jsonl        # noqa: E402
from core.video import frames as iter_frames          # noqa: E402
from identity.person_analysis import PersonAnalyzer, iou   # noqa: E402


def norm(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/seg_002")
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--ref-t", type=float, default=5.0)
    ap.add_argument("--thresh", type=float, default=0.62)
    args = ap.parse_args()

    d = Path(args.outdir)
    info = read_json(d / "video_info.json")
    fps = float(info["fps"])
    rows = {r["fi"]: r for r in read_jsonl(d / "target_tracks.jsonl")}
    if not rows:
        raise SystemExit("没有 target_tracks.jsonl")

    app = PersonAnalyzer()
    ref_fi = int(args.ref_t * fps)
    while ref_fi not in rows:
        ref_fi += 1

    ref_vecs = []
    for fi, frame in iter_frames(info["path"], skip=ref_fi, limit=1):
        r = rows[fi]
        tb = (r["x1"], r["y1"], r["x2"], r["y2"])
        for p in app.get(frame):
            if iou(p.bbox, tb) > 0.5 and p.reid is not None:
                ref_vecs.append(norm(p.reid))
    if not ref_vecs:
        raise SystemExit("参考帧没取到人体特征")
    ref = norm(np.mean(np.stack(ref_vecs), axis=0))
    print(f"基准: {args.ref_t}s 的目标人体特征")

    ts, sims, matched = [], [], 0
    total = 0
    for fi, frame in iter_frames(info["path"], limit=int(info["n_frames"])):
        if fi % args.every:
            continue
        total += 1
        r = rows.get(fi)
        if r is None:
            continue
        tb = (r["x1"], r["y1"], r["x2"], r["y2"])
        best = None
        for p in app.get(frame):
            if p.reid is None:
                continue
            v = iou(p.bbox, tb)
            if v > 0.45 and (best is None or v > best[0]):
                best = (v, p)
        if best is None:
            continue
        matched += 1
        ts.append(fi / fps)
        sims.append(float(ref @ norm(best[1].reid)))
    app.close()

    if not sims:
        raise SystemExit("没有可比对的样本")
    s = np.array(sims)
    bad = s < args.thresh
    print(f"统计 {len(s)} 个采样点（目标可见 {matched}/{total} 帧）")
    print(f"人体相似度 mean={s.mean():.3f} p10={np.percentile(s,10):.3f} "
          f"p50={np.median(s):.3f}")
    print(f"低于 {args.thresh}（疑似跟错人）: {bad.mean()*100:.1f}%")

    # 连续区间
    runs, cur = [], None
    t_all = np.array(ts)
    for i, b in enumerate(bad):
        if b and cur is None:
            cur = i
        elif not b and cur is not None:
            runs.append((cur, i - 1))
            cur = None
    if cur is not None:
        runs.append((cur, len(bad) - 1))
    runs = [r for r in runs if t_all[r[1]] - t_all[r[0]] >= 0.8]
    if runs:
        print("连续疑似区间：")
        for a, b in runs:
            print(f"  {t_all[a]:6.1f}s - {t_all[b]:6.1f}s "
                  f"({t_all[b]-t_all[a]:.1f}s, min={s[a:b+1].min():.2f})")
    else:
        print("没有连续性的疑似区间")


if __name__ == "__main__":
    main()
