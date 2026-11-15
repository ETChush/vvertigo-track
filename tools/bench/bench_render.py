"""渲染路径对比：Python 裁剪 vs ffmpeg 内部裁剪（sendcmd 逐帧改 crop）。

目的不是出片，是量出"让 ffmpeg 自己裁"能快多少、以及结果是否一致。

sendcmd 的写法有个坑：命令文件里每行 `时间 crop x 值;`，时间是**秒**，而且
crop filter 必须在 sendcmd 之后声明，否则命令没有目标。
"""
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

V = "F:/Projects/Dance/测试视频/3-00.01.56.791-00.02.38.694.mp4"
CSV = "F:/Projects/Dance/out_fast4/3_00_01_56_791_00_02_38_694/ID1/3_00_01_56_791_00_02_38_694-ID1.csv"
OUT = Path("work/_bench_render")
OUT.mkdir(parents=True, exist_ok=True)


def load_poses():
    rows = [l.split(",") for l in Path(CSV).read_text(encoding="utf-8").split("\n")
            if l and not l.startswith("#")]
    head = {c: i for i, c in enumerate(rows[0])}
    body = rows[1:]
    out = []
    for r in body:
        out.append(dict(fi=int(r[head["frame"]]), t=float(r[head["time"]]),
                        x=float(r[head["x"]]), cw=int(r[head["crop_w"]]),
                        ch=int(r[head["crop_h"]])))
    return out


poses = load_poses()
print(f"poses {len(poses)}", flush=True)
cw, ch = poses[0]["cw"], poses[0]["ch"]

# ---------------- A: ffmpeg 内部裁剪（sendcmd）
cmd_file = OUT / "cmd.txt"
lines = []
for p in poses:
    x0 = max(0, min(int(round(p["x"] - cw / 2)), 3840 - cw))
    lines.append(f"{p['t']:.6f} crop x {x0};")
cmd_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

argv = ["ffmpeg", "-v", "error", "-y", "-i", V,
        "-vf", f"sendcmd=f={cmd_file.name},crop={cw}:{ch}:0:0,scale=1080:1920",
        "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20", "-pix_fmt", "yuv420p",
        "-an", "a.mp4"]
t = time.time()
r = subprocess.run(argv, cwd=str(OUT), capture_output=True)
dt = time.time() - t
print(f"A 内部裁剪: {r.returncode} {dt:.1f}s = {len(poses) / dt:.1f} FPS", flush=True)
if r.returncode != 0:
    print(r.stderr.decode("utf-8", "ignore")[-400:], flush=True)

# 校验：抽 6 帧和"按 CSV 裁源帧"比 PSNR
if (OUT / "a.mp4").exists():
    src = cv2.VideoCapture(V)
    a = cv2.VideoCapture(str(OUT / "a.mp4"))
    ps = []
    for k in range(0, len(poses), max(1, len(poses) // 6)):
        p = poses[k]
        src.set(cv2.CAP_PROP_POS_FRAMES, p["fi"])
        ok, f = src.read()
        a.set(cv2.CAP_PROP_POS_FRAMES, p["fi"])
        ok2, g = a.read()
        if not (ok and ok2):
            continue
        x0 = max(0, min(int(round(p["x"] - cw / 2)), f.shape[1] - cw))
        crop = cv2.resize(f[0:ch, x0:x0 + cw], (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
        mse = np.mean((crop.astype(np.float32) - g.astype(np.float32)) ** 2)
        ps.append(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
    print("  PSNR vs 按 CSV 裁源帧: " + " ".join(f"{v:.1f}" for v in ps) + " dB", flush=True)

# ---------------- B: 现有 Python 裁剪路径（对照）
sys.path.insert(0, ".")
from core.fastvideo import pipe_frames  # noqa: E402

t = time.time()
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                        "-s", "1080x1920", "-r", "60", "-i", "-",
                        "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20",
                        "-pix_fmt", "yuv420p", "-an", "b.mp4"],
                       cwd=str(OUT), stdin=subprocess.PIPE)
n = 0
for fi, frame in pipe_frames(V):
    if n >= len(poses):
        break
    p = poses[n]
    x0 = max(0, min(int(round(p["x"] - cw / 2)), frame.shape[1] - cw))
    crop = frame[0:ch, x0:x0 + cw]
    if crop.shape[0] != ch or crop.shape[1] != cw:
        pad = np.zeros((ch, cw, 3), np.uint8)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad
    enc.stdin.write(cv2.resize(crop, (1080, 1920), interpolation=cv2.INTER_LINEAR).tobytes())
    n += 1
enc.stdin.close()
enc.wait()
dt = time.time() - t
print(f"B Python 裁剪: {dt:.1f}s = {n / dt:.1f} FPS", flush=True)
