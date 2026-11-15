# -*- coding: utf-8 -*-
"""全局身份预览：在一几帧上画出所有 global ID，供人工确认"我要跟几号"。

  python tools/preview_identities.py --outdir outputs/seg_002
  # 然后
  python main.py --stage target,plan,render --set target.global_id=5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import read_json, read_jsonl    # noqa: E402

PALETTE = [(60, 220, 255), (120, 255, 120), (255, 160, 80), (255, 120, 200),
           (200, 160, 255), (120, 220, 220), (255, 220, 120), (180, 180, 255)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/seg_002")
    ap.add_argument("--times", default="5,30,60,90")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = Path(args.outdir)
    info = read_json(d / "video_info.json")
    fps = float(info["fps"])
    local = {(r["sid"], r["lid"]): r for r in read_json(d / "local_tracks.json")}

    rows: dict[int, dict] = {}
    for r in read_jsonl(d / "tracks.jsonl"):
        gid = local.get((r["sid"], r["lid"]), {}).get("gid", -1)
        if gid and gid > 0:
            rows.setdefault(r["fi"], {})[gid] = r

    print(f"{'gid':>4} {'帧数':>6} {'起止(s)':>14} {'身高':>6} {'人脸样本':>7}")
    agg: dict[int, dict] = {}
    for k, r in local.items():
        g = int(r["gid"])
        if g <= 0:
            continue
        a = agg.setdefault(g, dict(n=0, lo=1e9, hi=-1, h=[], fn=0))
        a["n"] += r["n"]
        a["lo"] = min(a["lo"], r["start"])
        a["hi"] = max(a["hi"], r["end"])
        a["h"].append(r["h"])
        a["fn"] += r["face_n"]
    for g, a in sorted(agg.items(), key=lambda x: -x[1]["n"]):
        print(f"{g:>4} {a['n']:>6} {a['lo']/fps:6.1f}-{a['hi']/fps:6.1f}s "
              f"{np.median(a['h']):>6.0f} {a['fn']:>7}")

    cap = cv2.VideoCapture(info["path"])
    tiles = []
    for t in [float(x) for x in args.times.split(",") if x.strip()]:
        fi = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        for gid, r in rows.get(fi, {}).items():
            c = PALETTE[gid % len(PALETTE)]
            cv2.rectangle(frame, (int(r["x1"]), int(r["y1"])), (int(r["x2"]), int(r["y2"])), c, 3)
            cv2.putText(frame, f"ID {gid}", (int(r["x1"]) + 4, int(r["y1"]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{t:.0f}s", (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (255, 255, 255), 3, cv2.LINE_AA)
        tiles.append(cv2.resize(frame, (760, 428), interpolation=cv2.INTER_AREA))
    cap.release()

    if not tiles:
        raise SystemExit("没取到帧")
    cols = 2
    rows_n = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((428 * rows_n, 760 * cols, 3), np.uint8)
    for k, im in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * 428:(r + 1) * 428, c * 760:(c + 1) * 760] = im

    p = Path(args.out) if args.out else d / "identity_preview.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"\n预览图: {p}")


if __name__ == "__main__":
    main()
