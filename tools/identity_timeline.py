# -*- coding: utf-8 -*-
"""身份一致性时间线：以参考人脸为锚，逐段检查成片里被框的人是否还是同一个人。

这是"跟错人"最直接的证据图——曲线掉下去的地方，就是换了人的地方。

  python tools/identity_timeline.py --dirs outputs/seg_002 outputs/taq600 --ref work/ref_face.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import read_json, read_jsonl        # noqa: E402
from core.video import frames as iter_frames          # noqa: E402
from identity.face import FaceEncoder, assign_to_tracks  # noqa: E402


def collect(outdir: Path, ref: np.ndarray, enc: FaceEncoder, every: int = 15):
    info = read_json(outdir / "video_info.json")
    rows = {r["fi"]: r for r in read_jsonl(outdir / "target_tracks.jsonl")}
    ts, sims = [], []
    for fi, frame in iter_frames(info["path"], limit=int(info["n_frames"])):
        if fi % every:
            continue
        r = rows.get(fi)
        if r is None:
            continue
        faces = enc.embed(frame)
        if not faces:
            continue
        hit = assign_to_tracks(faces, [((0, 0), (r["x1"], r["y1"], r["x2"], r["y2"]))])
        f = hit.get((0, 0))
        if f is None:
            continue
        ts.append(fi / float(info["fps"]))
        sims.append(float(ref @ f.feat))
    return np.array(ts), np.array(sims)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--out", default="outputs/identity_timeline.png")
    args = ap.parse_args()

    cache = Path(args.out).with_suffix(".npz")
    enc = None
    ref = None
    series = []
    for d in args.dirs:
        key = f"t_{Path(d).name}"
        val = f"s_{Path(d).name}"
        if cache.exists():
            z = np.load(cache)
            if key in z and val in z:
                t, s = z[key], z[val]
                print(f"{Path(d).name}: 复用缓存 {len(s)} 帧 mean={s.mean():.3f}")
                series.append((Path(d).name, t, s))
                continue
        if enc is None:
            enc = FaceEncoder()
            ref = enc.embed_file(args.ref)
        t, s = collect(Path(d), ref, enc, args.every)
        print(f"{Path(d).name}: {len(s)} 帧有脸，mean={s.mean():.3f} "
              f"p10={np.percentile(s,10):.3f}  <0.5 {(s<0.5).mean()*100:.1f}%")
        series.append((Path(d).name, t, s))

    if not series:
        return
    np.savez(cache, **{f"t_{n}": t for n, t, _ in series},
             **{f"s_{n}": v for n, _, v in series})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:       # noqa: BLE001
        print(f"matplotlib 不可用（{exc}），跳过绘图")
        return

    fig, ax = plt.subplots(figsize=(13, 4.2), dpi=140)
    colors = ["#2f80ed", "#eb5757", "#27ae60"]
    for i, (name, t, s) in enumerate(series):
        ax.plot(t, s, lw=1.1, alpha=0.85, color=colors[i % 3], label=name)
        ax.axhline(s.mean(), color=colors[i % 3], ls=":", lw=0.8, alpha=0.6)
    ax.axhline(0.5, color="#888", ls="--", lw=1.0)
    ax.text(1, 0.51, "wrong-person line 0.50", fontsize=8, color="#666")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("similarity to reference face")
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlim(0, max(t[-1] for _, t, _ in series))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("Identity consistency: is the framed person still the same dancer?")
    fig.tight_layout()
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    print(f"时间线: {p}")


if __name__ == "__main__":
    main()
