# -*- coding: utf-8 -*-
"""定位"从哪一帧开始跟错人"。

做法：以参考人脸为锚，逐段算被框住那个人与锚的相似度，再做 3 秒中值平滑
（去掉单帧噪声），输出连续低于阈值的区间，并把可疑帧的人物裁图存下来备查。

  python tools/locate_drift.py --outdir outputs/seg_002 --ref work/ref_face.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import read_json, read_jsonl        # noqa: E402
from core.video import frames as iter_frames          # noqa: E402
from identity.face import FaceEncoder, assign_to_tracks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/seg_002")
    ap.add_argument("--ref", default="work/ref_face.jpg")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--min-run", type=float, default=2.0, help="最短可疑时长（秒）")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--dump", default="work/drift_frames")
    args = ap.parse_args()

    d = Path(args.outdir)
    info = read_json(d / "video_info.json")
    fps = float(info["fps"])
    rows = {r["fi"]: r for r in read_jsonl(d / "target_tracks.jsonl")}
    enc = FaceEncoder()
    ref = enc.embed_file(args.ref)

    ts, sims, has = [], [], []
    for fi, frame in iter_frames(info["path"], limit=int(info["n_frames"])):
        if fi % args.every:
            continue
        r = rows.get(fi)
        if r is None:
            continue
        faces = enc.embed(frame)
        hit = assign_to_tracks(faces, [((0, 0), (r["x1"], r["y1"], r["x2"], r["y2"]))]) \
            if faces else {}
        f = hit.get((0, 0))
        ts.append(fi / fps)
        sims.append(float(ref @ f.feat) if f is not None else np.nan)
        has.append(f is not None)

    ts = np.array(ts); sims = np.array(sims)
    ok = ~np.isnan(sims)
    # 3 秒中值平滑（每侧 ~3/(every/fps) 个样本）
    win = max(3, int(round(3.0 / (args.every / fps))) | 1)
    sm = np.full_like(sims, np.nan)
    for i in range(len(sims)):
        a, b = max(0, i - win // 2), min(len(sims), i + win // 2 + 1)
        seg = sims[a:b]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= 2:
            sm[i] = float(np.median(seg))

    print(f"采样 {ok.sum()} 帧有脸 / {len(sims)} 帧目标可见")
    print(f"平滑后: mean={np.nanmean(sm):.3f}  低于 {args.thresh} 的采样点 "
          f"{np.nanmean(sm < args.thresh)*100:.1f}%  最短时长过滤 {args.min_run}s")

    runs = []
    cur = None
    for i, v in enumerate(sm):
        bad = (not np.isnan(v)) and v < args.thresh
        if bad and cur is None:
            cur = i
        elif not bad and cur is not None:
            runs.append((cur, i - 1))
            cur = None
    if cur is not None:
        runs.append((cur, len(sm) - 1))
    runs = [r for r in runs if ts[r[1]] - ts[r[0]] >= args.min_run]
    if not runs:
        print("没有发现持续性的身份异常")
        return

    print("\n持续可疑区间：")
    out_dir = Path(args.dump)
    out_dir.mkdir(parents=True, exist_ok=True)
    for a, b in runs:
        seg = sm[a:b + 1]
        print(f"  {ts[a]:6.1f}s - {ts[b]:6.1f}s  ({ts[b]-ts[a]:.1f}s)  "
              f"平滑相似度 {np.nanmin(seg):.3f}~{np.nanmax(seg):.3f}")
        t_mid = (ts[a] + ts[b]) / 2
        fi = int(t_mid * fps)
        frame = None
        for f_i, fr in iter_frames(info["path"], skip=fi, limit=1):
            frame = fr
        if frame is not None and fi in rows:
            r = rows[fi]
            pad = 0.3
            w, h = r["x2"] - r["x1"], r["y2"] - r["y1"]
            x0 = int(max(0, r["x1"] - w * pad)); y0 = int(max(0, r["y1"] - h * pad))
            x1 = int(min(frame.shape[1], r["x2"] + w * pad)); y1 = int(min(frame.shape[0], r["y2"] + h * pad))
            crop = frame[y0:y1, x0:x1]
            p = out_dir / f"drift_{t_mid:05.1f}s.jpg"
            cv2.imwrite(str(p), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            print(f"     裁图: {p}")


if __name__ == "__main__":
    main()
