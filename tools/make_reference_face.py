# -*- coding: utf-8 -*-
"""从某一帧的人脸生成参考人脸图，用于跨运行锁定同一个人。

  # 先看预览：哪一帧有清晰正脸
  python tools/make_reference_face.py --at 5.0 --list
  # 选一个脸（按点或序号），导出对齐后的 112x112
  python tools/make_reference_face.py --at 5.0 --index 2 --out work/ref_face.jpg
  # 或者直接给点坐标
  python tools/make_reference_face.py --at 5.0 --point 1180,300 --out work/ref_face.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.video import probe, read_frame_at     # noqa: E402
from identity.face import FaceEncoder          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="seg_002.mp4")
    ap.add_argument("--at", type=float, default=5.0)
    ap.add_argument("--index", type=int, default=-1, help="第几个脸（按检测顺序，左→右）")
    ap.add_argument("--point", default="", help="人脸内的任意点 x,y")
    ap.add_argument("--out", default="work/ref_face.jpg")
    ap.add_argument("--list", action="store_true", help="只列出检测到的人脸")
    args = ap.parse_args()

    info = probe(args.video)
    frame = read_frame_at(args.video, int(args.at * info.fps))
    if frame is None:
        raise SystemExit("取帧失败")

    enc = FaceEncoder()
    faces = enc.embed(frame)
    if not faces:
        raise SystemExit(f"{args.at}s 这一帧没有检出人脸，换个时间点")

    faces.sort(key=lambda f: f.cx)
    print(f"{args.at}s 检出 {len(faces)} 张脸：")
    for i, f in enumerate(faces):
        print(f"  #{i}  中心=({f.cx:.0f},{f.cy:.0f})  高={f.h:.0f}px  检测分={f.score:.2f}")

    if args.list:
        vis = frame.copy()
        for i, f in enumerate(faces):
            cv2.rectangle(vis, (int(f.x1), int(f.y1)), (int(f.x2), int(f.y2)), (0, 255, 200), 3)
            cv2.putText(vis, f"#{i}", (int(f.x1), int(f.y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2, cv2.LINE_AA)
        p = Path("work/face_candidates.jpg")
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), vis)
        print(f"预览图: {p}")
        return

    target = None
    if args.point:
        px, py = (float(v) for v in args.point.split(","))
        target = min(faces, key=lambda f: (f.cx - px) ** 2 + (f.cy - py) ** 2)
    if target is None:
        if not (0 <= args.index < len(faces)):
            raise SystemExit("请用 --index 或 --point 指定一张脸")
        target = faces[args.index]

    # 用识别器的对齐流程导出 112x112，保证与后续 embed_file 一致
    det = enc.det
    det.setInputSize((frame.shape[1], frame.shape[0]))
    _, raw = det.detect(frame)
    best_row, best_d = None, 1e9
    for row in raw:
        cx, cy = row[0] + row[2] / 2, row[1] + row[3] / 2
        d = (cx - target.cx) ** 2 + (cy - target.cy) ** 2
        if d < best_d:
            best_row, best_d = row, d
    aligned = enc.rec.alignCrop(frame, best_row)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), aligned)
    print(f"参考人脸已保存: {p}  ({aligned.shape[1]}x{aligned.shape[0]})")
    print(f"用法: --set target.face_image={p.as_posix()}")


if __name__ == "__main__":
    main()
