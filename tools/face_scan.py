"""全片人脸扫描：逐帧在所有检测框上比对参考人脸，输出每个时刻"她是否在画面里"。

用途不是出片，是量出这条素材的物理上限：
  - 她在画面里的真实时长占比（人脸可见率）
  - 哪条 TAQ 轨迹在哪个时段其实是她（用于判断轨迹串人）
  - 各时段的最高相似度，看阈值该定多少

只读不写，除了一个 json 报告。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.video import open_capture                      # noqa: E402
from identity.face import FaceEncoder                    # noqa: E402


def load_ref(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不到参考图：{path}")
    enc = FaceEncoder()
    obs = enc.embed(img)
    if not obs:
        return np.asarray(enc.embed_aligned(img), dtype=np.float32)
    return np.asarray(max(obs, key=lambda o: o.score).feat, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True, help="tracks_kept.jsonl")
    ap.add_argument("--ref", required=True, help="参考人脸图")
    ap.add_argument("--out", required=True, help="输出 json")
    ap.add_argument("--every", type=int, default=15, help="采样间隔（帧）")
    ap.add_argument("--topk", type=int, default=10, help="每帧最多检查几个框（按高度）")
    args = ap.parse_args()

    byf: dict[int, list] = defaultdict(list)
    for line in open(args.tracks, encoding="utf-8"):
        r = json.loads(line)
        byf[r["fi"]].append(r)
    want = {fi for fi in byf if fi % args.every == 0}
    print(f"[scan] 采样 {len(want)} 帧 / 共 {len(byf)} 个有检测的帧", flush=True)

    ref = load_ref(args.ref)
    enc = FaceEncoder()
    cap = open_capture(args.video)
    results = []
    t0 = time.time()
    fi = 0
    while True:
        if fi in want:
            ok, frame = cap.read()
            if not ok:
                break
            bs = sorted(byf[fi], key=lambda r: -(r["y2"] - r["y1"]))[: args.topk]
            best = (-1.0, None, None, None)
            for r in bs:
                b = (r["x1"], r["y1"], r["x2"], r["y2"])
                obs = enc.embed_head(frame, b)
                if not obs:
                    continue
                o = max(obs, key=lambda x: x.score)
                s = float(o.feat @ ref)
                if s > best[0]:
                    best = (s, int(r["gid"]), b, float(o.h))
            results.append(dict(fi=fi, n=len(bs), sim=round(best[0], 4),
                                gid=best[1], face_h=(None if best[3] is None else round(best[3], 1)),
                                cx=None if best[2] is None else round((best[2][0] + best[2][2]) / 2, 1)))
            if len(results) % 100 == 0:
                el = time.time() - t0
                print(f"[scan] {len(results)}/{len(want)} 帧  {el:.0f}s", flush=True)
        elif not cap.grab():
            break
        fi += 1
    cap.release()

    sims = np.array([r["sim"] for r in results])
    print(f"\n[scan] 完成 {len(results)} 帧，{(time.time() - t0) / 60:.1f} 分钟")
    for thr in (0.65, 0.60, 0.55, 0.50, 0.45):
        hit = int((sims >= thr).sum())
        print(f"  相似度 >= {thr}: {hit}/{len(sims)} = {hit / max(1, len(sims)) * 100:.1f}%")

    hits = [r for r in results if r["sim"] >= 0.55]
    if hits:
        gcnt = defaultdict(int)
        for r in hits:
            gcnt[r["gid"]] += 1
        print("\n  命中帧里各 gid 的占比：")
        for g, n in sorted(gcnt.items(), key=lambda x: -x[1])[:8]:
            print(f"    gid{g:3d}: {n:5d} 帧 ({n / len(hits) * 100:.1f}%)")

        # 可见段
        segs, s, p = [], hits[0]["fi"], hits[0]["fi"]
        for r in hits[1:]:
            if r["fi"] > p + args.every * 3:
                segs.append((s, p))
                s = r["fi"]
            p = r["fi"]
        segs.append((s, p))
        fps = 59.9401
        print(f"\n  可见段（间隔 >{args.every * 3} 帧算断开）:")
        for a, b in segs:
            if (b - a) / args.every < 2:
                continue
            print(f"    {a / fps:7.1f}-{b / fps:7.1f}s  n={int((b - a) / args.every) + 1}")

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"\n[scan] 写入 {args.out}")


if __name__ == "__main__":
    main()
