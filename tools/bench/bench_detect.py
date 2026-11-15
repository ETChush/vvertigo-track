"""检测与人脸的吞吐基准。

检测用 1280 矩形帧（和 imgsz=1280 对齐，省掉 YOLO 内部的二次缩放），
比较 batch=1 和 batch=8/16 的差异。人脸按"每帧 top-k 个框各裁一次头"计。
"""
import subprocess
import time

import cv2
import numpy as np

V = "F:/Projects/Dance/测试视频/3-00.01.56.791-00.02.38.694.mp4"


def frames_1280(n):
    cmd = ["ffmpeg", "-v", "error", "-i", V, "-vf", "scale=1280:720",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1024 * 1024)
    out = []
    buf = np.empty((720, 1280, 3), np.uint8)
    view = buf.reshape(-1)
    for _ in range(n):
        if p.stdout.readinto(view) != view.nbytes:
            break
        out.append(buf.copy())
    p.stdout.close()
    p.kill()
    p.wait()
    return out


def frames_4k(n):
    cmd = ["ffmpeg", "-v", "error", "-i", V, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1024 * 1024)
    out = []
    buf = np.empty((2160, 3840, 3), np.uint8)
    view = buf.reshape(-1)
    for _ in range(n):
        if p.stdout.readinto(view) != view.nbytes:
            break
        out.append(buf.copy())
    p.stdout.close()
    p.kill()
    p.wait()
    return out


print("取帧...", flush=True)
F1280 = frames_1280(140)
F4K = frames_4k(12)
print(f"  1280: {len(F1280)} 帧   4K: {len(F4K)} 帧", flush=True)

from ultralytics import YOLO  # noqa: E402

w = "yolo11m.pt"
import glob  # noqa: E402

cands = glob.glob("**/yolo11m.pt", recursive=True)
if cands:
    w = cands[0]
print(f"权重 {w}", flush=True)
model = YOLO(w)


def bench(imgs, imgsz, batch, tag, half=True):
    kw = dict(imgsz=imgsz, conf=0.25, iou=0.6, classes=[0], device=0,
              verbose=False, half=half)
    model.predict(imgs[0], **kw)          # warmup
    t = time.time()
    n, nb = 0, 0
    while n + batch <= len(imgs):
        model.predict(imgs[n:n + batch], **kw)
        n += batch
        nb += 1
    dt = time.time() - t
    print(f"  {tag:34s} {n:4d} 帧 {dt:5.2f}s = {n / dt:6.1f} FPS", flush=True)
    return n / dt


print("YOLO 检测：", flush=True)
bench(F1280[:32], 1280, 1, "1280 batch1")
bench(F1280[:96], 1280, 8, "1280 batch8")
bench(F1280[:128], 1280, 16, "1280 batch16")
bench(F1280[:96], 960, 8, "960  batch8")
bench(F1280[:32], 1280, 1, "1280 batch1 fp32", half=False)

print("人脸（SFace embed_head，逐框）：", flush=True)
import sys  # noqa: E402

sys.path.insert(0, ".")
from identity.face import FaceEncoder  # noqa: E402

enc = FaceEncoder()
sub = F4K[:6]
# 用 1280 检测框按比例放大成 4K 框，模拟真实流程
model.predict(sub[0][::3, ::3], imgsz=1280, conf=0.25, classes=[0], device=0, verbose=False)
t = time.time()
nb = 0
tot = 0
for f in sub:
    r = model.predict(f[::3, ::3], imgsz=1280, conf=0.25, classes=[0], device=0,
                      verbose=False)[0]
    if r.boxes is None:
        continue
    for b in r.boxes.xyxy.cpu().numpy()[:8]:
        box = tuple(float(v) * 3 for v in b)
        enc.embed_head(f, box)
        nb += 1
    tot += 1
dt = time.time() - t
print(f"  {tot} 帧 / {nb} 个头部  {dt:5.2f}s = {nb / dt:6.1f} 头/秒  "
      f"({tot / dt:5.1f} 帧/秒)", flush=True)
