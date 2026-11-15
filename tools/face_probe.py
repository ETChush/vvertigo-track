# -*- coding: utf-8 -*-
"""人脸可用性探针：在这段素材上量化「人脸能不能当身份锚点」。

三个问题一次答完：
  1. 人脸有多大（决定能不能上识别模型）
  2. 同一个人跨帧的相似度 vs 不同人之间的相似度（决定有没有区分度）
  3. 检测/识别耗时

  python tools/face_probe.py --video seg_002.mp4 --tracks outputs/seg_002/tracks.jsonl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import read_jsonl                      # noqa: E402
from core.video import frames as iter_frames, probe     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="seg_002.mp4")
    ap.add_argument("--tracks", default="outputs/seg_002/tracks.jsonl")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--min-h", type=float, default=250, help="只统计这个高度以上的人（舞者）")
    ap.add_argument("--out", default="outputs/face_probe.json")
    args = ap.parse_args()

    info = probe(args.video)
    f0, f1 = int(args.t0 * info.fps), int(args.t1 * info.fps)

    # 舞者的局部轨迹
    by_track: dict[int, list] = {}
    for r in read_jsonl(args.tracks):
        if not (f0 <= r["fi"] < f1):
            continue
        if r["y2"] - r["y1"] < args.min_h:
            continue
        by_track.setdefault(r["lid"], []).append(r)

    det = cv2.FaceDetectorYN.create("work/models/yunet.onnx", "", (info.width, info.height),
                                    0.6, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create("work/models/sface.onnx", "")

    feats: dict[int, list[np.ndarray]] = {k: [] for k in by_track}
    sizes: list[float] = []
    t_det = t_rec = 0.0
    n_face = 0

    for fi, frame in iter_frames(args.video, limit=f1):
        if fi < f0 or fi % args.step:
            continue
        det.setInputSize((frame.shape[1], frame.shape[0]))
        a = time.time()
        _, faces = det.detect(frame)
        t_det += time.time() - a
        if faces is None:
            continue
        for f in faces:
            fx, fy, fw, fh = f[0], f[1], f[2], f[3]
            if fh < 32:
                continue
            cx, cy = fx + fw / 2, fy + fh / 2
            best, best_d = None, 1e9
            for lid, rows in by_track.items():
                r = next((r for r in rows if r["fi"] == fi), None)
                if r is None:
                    continue
                if not (r["x1"] <= cx <= r["x2"] and r["y1"] <= cy <= r["y2"]):
                    continue
                d = abs((r["x1"] + r["x2"]) / 2 - cx)
                if d < best_d:
                    best, best_d = lid, d
            if best is None:
                continue
            b = time.time()
            aligned = rec.alignCrop(frame, f)
            feat = rec.feature(aligned).flatten()
            t_rec += time.time() - b
            n_face += 1
            sizes.append(float(fh))
            feats[best].append(feat / (np.linalg.norm(feat) + 1e-9))

    feats = {k: v for k, v in feats.items() if len(v) >= 3}
    keys = sorted(feats, key=lambda k: -len(feats[k]))
    print(f"t={args.t0}-{args.t1}s  舞者轨迹 {len(by_track)} 条，可用人脸 {n_face} 个，"
          f"覆盖 {len(keys)} 条轨迹")
    if sizes:
        s = np.array(sizes)
        print(f"人脸高度 p10/p50/p90 = {np.percentile(s,10):.0f}/"
              f"{np.percentile(s,50):.0f}/{np.percentile(s,90):.0f} px，"
              f"≥64px 占 {(s>=64).mean()*100:.1f}%")
        print(f"单帧耗时：检测 {t_det/max(1,len(sizes))*1000:.1f}ms  "
              f"识别 {t_rec/max(1,n_face)*1000:.1f}ms")
    if len(keys) < 2:
        print("轨迹太少，无法比较")
        return

    means = {k: np.mean(np.stack(feats[k]), axis=0) for k in keys}
    means = {k: v / np.linalg.norm(v) for k, v in means.items()}

    print("\n同一人跨帧相似度（越大越稳）：")
    for k in keys:
        m = np.stack(feats[k])
        cen = np.mean(m, axis=0)
        cen /= np.linalg.norm(cen)
        sims = m @ cen
        print(f"  轨迹 {k}: {len(m):3d} 个样本  mean={sims.mean():.3f} p10={np.percentile(sims,10):.3f}")

    print("\n不同人之间相似度（越小越好分）：")
    vals = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            s = float(means[keys[i]] @ means[keys[j]])
            vals.append(s)
            print(f"  轨迹 {keys[i]:>3} vs {keys[j]:>3}: {s:.3f}")
    if vals:
        print(f"  跨人 mean={np.mean(vals):.3f} max={np.max(vals):.3f}")


if __name__ == "__main__":
    main()
