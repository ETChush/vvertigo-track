# -*- coding: utf-8 -*-
"""成片身份自检：目标框里的人脸，是否始终匹配同一个身份库。

这是目前唯一能自动发现"跟错人"的客观检查——不需要人工标注，也不依赖跟踪器
自己的说法。

  python tools/verify_target_identity.py --outdir outputs/seg_002
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import read_json, read_jsonl          # noqa: E402
from core.video import frames as iter_frames          # noqa: E402
from identity.face import FaceEncoder, assign_to_tracks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/seg_002")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--gallery", type=int, default=6)
    ap.add_argument("--ref", default="", help="参考人脸图；给了就直接以它为锚，公平跨运行比较")
    args = ap.parse_args()

    d = Path(args.outdir)
    info = read_json(d / "video_info.json")
    rows = {r["fi"]: r for r in read_jsonl(d / "target_tracks.jsonl")}
    enc = FaceEncoder()

    gal: list[np.ndarray] = []
    if args.ref:
        gal.append(enc.embed_file(args.ref))
    sims: list[tuple[int, float, float]] = []
    n_face = 0

    for fi, frame in iter_frames(info["path"], limit=int(info["n_frames"])):
        if fi % args.every:
            continue
        r = rows.get(fi)
        if r is None:
            continue
        box = (r["x1"], r["y1"], r["x2"], r["y2"])
        faces = enc.embed(frame)
        hit = assign_to_tracks(faces, [((0, 0), box)])
        f = hit.get((0, 0))
        if f is None:
            continue
        n_face += 1
        if len(gal) < args.gallery and not args.ref:
            gal.append(f.feat)
            continue
        g = np.mean(np.stack(gal), axis=0)
        g /= np.linalg.norm(g) + 1e-9
        sims.append((fi, float(f.feat @ g), f.score))

    if not sims:
        print("有效人脸不足，无法自检")
        return

    s = np.array([x[1] for x in sims])
    print(f"抽查 {len(sims)} 帧含目标人脸（共检测 {n_face} 帧有脸）")
    print(f"与人脸库相似度: mean={s.mean():.3f} p10={np.percentile(s,10):.3f} "
          f"p50={np.median(s):.3f} min={s.min():.3f}")
    for thr in (0.55, 0.60, 0.65):
        print(f"  >= {thr}: {(s >= thr).mean()*100:.1f}%")
    bad = [x for x in sims if x[1] < 0.5]
    print(f"疑似错人帧（<0.5）: {len(bad)} 帧 ({len(bad)/len(sims)*100:.1f}%)")
    for fi, v, sc in bad[:10]:
        print(f"   frame {fi} ({fi/info['fps']:.1f}s) sim={v:.3f} det={sc:.2f}")


if __name__ == "__main__":
    main()
