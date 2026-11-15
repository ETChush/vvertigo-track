# -*- coding: utf-8 -*-
"""裁切几何对比：旧 fancam 的 pad 缩放 vs 满幅裁切。

旧逻辑（Dance11.8/app/media.py: crop_window）按「身高 × pad」定窗口高度，
拿 pad=1.30、身高 500px 的舞者算，窗口只有 360×640，要放大 3 倍才能铺满
1080×1920。这里把两种裁切都铺满同一输出尺寸，直接看清晰度差多少。

  python tools/compare_crop_scale.py --at 30 --box x1,y1,x2,y2
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

OUT_W, OUT_H = 1080, 1920


def old_crop(box, src_w, src_h, pad=1.30):
    """复刻 Dance11.8 的 crop_window：按身高×pad 定边，18k×32k 精确 9:16。

    注意入参是 (x1, y1, x2, y2)；原函数用的是 (x, y, w, h)，这里先换算。
    """
    x1, y1, x2, y2 = box
    bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
    target_h = bh * pad
    k = int(round(target_h / 32.0))
    k = max(1, min(k, src_h // 32, src_w // 18))
    ch, cw = 32 * k, 18 * k
    foot = by + bh
    top = foot - 0.88 * ch if bh <= 0.90 * src_h else by - 0.08 * ch
    left = (bx + bw / 2.0) - cw / 2.0
    left = max(0.0, min(left, src_w - cw))
    top = max(0.0, min(top, src_h - ch))
    return int(left), int(top), cw, ch


def new_crop(box, src_w, src_h, aspect=(9, 16)):
    cw = int(round(src_h * aspect[0] / aspect[1]))
    cx = (box[0] + box[2]) / 2.0
    left = int(round(max(0, min(cx - cw / 2.0, src_w - cw))))
    return left, 0, cw, src_h


def upscale(img, w=OUT_W, h=OUT_H, flags=cv2.INTER_LANCZOS4):
    return cv2.resize(img, (w, h), interpolation=flags)


def label(img, text, scale=1.4):
    im = img.copy()
    cv2.rectangle(im, (0, 0), (int(28 * len(text) * scale / 1.4), 54), (0, 0, 0), -1)
    cv2.putText(im, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="seg_002.mp4")
    ap.add_argument("--at", type=float, default=30.0)
    ap.add_argument("--box", required=True, help="人体框 x1,y1,x2,y2")
    ap.add_argument("--out", default="outputs/crop_compare.jpg")
    args = ap.parse_args()

    box = [float(v) for v in args.box.split(",")]
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.at * 60))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("取帧失败")
    H, W = frame.shape[:2]

    lx, ly, lw, lh = old_crop(box, W, H)
    a = upscale(frame[ly:ly + lh, lx:lx + lw])
    nx, ny, nw, nh = new_crop(box, W, H)
    b = upscale(frame[ny:ny + nh, nx:nx + nw])

    a = label(a, f"OLD  {lw}x{lh} -> {OUT_W}x{OUT_H}  ({OUT_W/lw:.2f}x)", 1.0)
    b = label(b, f"NEW  {nw}x{nh} -> {OUT_W}x{OUT_H}  ({OUT_W/nw:.2f}x)", 1.0)

    # 对齐到被摄者的头部/上半身，各自从成品里原样抠出来比较（这是唯一公平的比法）
    def head_patch(img, cx, cy, cw, ch, box, size=320):
        sx = OUT_W / cw
        bx = (box[0] - cx) * sx
        bw = (box[2] - box[0]) * sx
        by = (box[1] - cy) * sx
        bh = (box[3] - box[1]) * sx
        x0 = int(max(0, bx - bw * 0.12)); y0 = int(max(0, by - bh * 0.08))
        x1 = int(min(img.shape[1], bx + bw * 1.12)); y1 = int(min(img.shape[0], by + bh * 0.45))
        patch = img[y0:y1, x0:x1]
        return cv2.resize(patch, (size, size), interpolation=cv2.INTER_NEAREST)

    ca = head_patch(a, lx, ly, lw, lh, box)
    cb = head_patch(b, nx, ny, nw, nh, box)
    detail = np.hstack([ca, cb])
    detail = label(detail, f"OLD {OUT_W/lw:.2f}x  |  NEW {OUT_W/nw:.2f}x", 0.7)

    sheet = np.hstack([a, b])
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    p2 = p.with_name(p.stem + "_detail.jpg")
    cv2.imwrite(str(p2), detail, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"旧裁切 {lw}x{lh} 放大 {OUT_W/lw:.2f}x | 新裁切 {nw}x{nh} 放大 {OUT_W/nw:.2f}x")
    print(f"有效像素面积比 {(nw*nh)/(lw*lh):.2f}x")
    print(f"对比图 {p}\n细节图 {p2}")


if __name__ == "__main__":
    main()
