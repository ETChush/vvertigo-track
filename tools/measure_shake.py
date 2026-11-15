# -*- coding: utf-8 -*-
"""从成片本身测抖动：逐帧相位相关求位移，再看 0.8-4Hz 频段的幅度。

直接测像素，能覆盖"路径平滑了但渲染取整又抖回去"这类问题。
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def measure(path, t0, t1, fps=60.0, scale=0.25):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    prev = None
    dx = []
    hann = cv2.createHanningWindow((160, 90), cv2.CV_32F)
    n = 0
    while n < int((t1 - t0) * fps):
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA)
        if prev is not None:
            (sx, sy), resp = cv2.phaseCorrelate(prev, g, hann)
            dx.append(sx / scale)      # 换算回 1080 宽画面的像素
        prev = g
        n += 1
    cap.release()
    return np.array(dx)


def band(y, lo_w=3, hi_w=30):
    return float((gaussian_filter1d(y, lo_w) - gaussian_filter1d(y, hi_w)).std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, default=40.0)
    ap.add_argument("--t1", type=float, default=75.0)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    print(f"逐帧位移（成片 1080 宽的像素），{args.t0:.0f}-{args.t1:.0f}s\n")
    print(f"{'文件':<40}{'位移std':>9}{'0.8-4Hz':>10}{'|d2|p95':>10}{'反转/秒':>9}")
    for f in args.files:
        d = measure(f, args.t0, args.t1)
        if len(d) < 10:
            print(f"{f:<40} (读取失败)")
            continue
        a = np.diff(d)
        sgn = np.sign(a[np.abs(a) > 0.3])
        rev = int(np.sum(sgn[1:] != sgn[:-1])) if len(sgn) > 1 else 0
        secs = len(d) / 60.0
        print(f"{f.split(chr(92))[-1].split('/')[-1]:<40}"
              f"{d.std():9.2f}{band(d):10.2f}{np.percentile(np.abs(a),95):10.2f}{rev/secs:9.2f}")


if __name__ == "__main__":
    main()
