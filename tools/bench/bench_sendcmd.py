"""测 sendcmd 的时间戳对齐：帧边界 / 帧中心 / 提前半帧。

只跑一次编码，用 8 个采样点算 PSNR，避免超时。
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

rows = [l.split(",") for l in Path(CSV).read_text(encoding="utf-8").split("\n")
        if l and not l.startswith("#")]
h = {c: i for i, c in enumerate(rows[0])}
body = rows[1:]
poses = [dict(fi=int(r[h["frame"]]), t=float(r[h["time"]]), x=float(r[h["x"]])) for r in body]
cw, ch = int(body[0][h["crop_w"]]), int(body[0][h["crop_h"]])
fps = 60000 / 1001

tag = sys.argv[1] if len(sys.argv) > 1 else "center"
tf = {
    "edge": lambda p: p["t"],
    "center": lambda p: (p["fi"] + 0.5) / fps,
    "nudge": lambda p: (p["fi"] + 0.02) / fps,
}[tag]

cmd = OUT / f"cmd_{tag}.txt"
cmd.write_text("\n".join(
    f"{tf(p):.6f} crop x {max(0, min(int(round(p['x'] - cw / 2)), 3840 - cw))};"
    for p in poses) + "\n", encoding="utf-8")

argv = ["ffmpeg", "-v", "error", "-y", "-i", V,
        "-vf", f"sendcmd=f={cmd.name},crop={cw}:{ch}:0:0,scale=1080:1920",
        "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20", "-pix_fmt", "yuv420p",
        "-an", f"{tag}.mp4"]
t = time.time()
r = subprocess.run(argv, cwd=str(OUT), capture_output=True)
dt = time.time() - t
print(f"{tag}: rc={r.returncode}  {len(poses) / dt:.1f} FPS  {dt:.1f}s", flush=True)
if r.returncode != 0:
    print(r.stderr.decode("utf-8", "ignore")[-300:], flush=True)
    raise SystemExit(1)

src = cv2.VideoCapture(V)
a = cv2.VideoCapture(str(OUT / f"{tag}.mp4"))
ps = []
for k in range(0, len(poses), max(1, len(poses) // 8)):
    p = poses[k]
    src.set(cv2.CAP_PROP_POS_FRAMES, p["fi"])
    ok, f = src.read()
    a.set(cv2.CAP_PROP_POS_FRAMES, p["fi"])
    ok2, g = a.read()
    if not (ok and ok2):
        continue
    x0 = max(0, min(int(round(p["x"] - cw / 2)), f.shape[1] - cw))
    c = cv2.resize(f[0:ch, x0:x0 + cw], (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
    mse = np.mean((c.astype(np.float32) - g.astype(np.float32)) ** 2)
    ps.append(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
print("  PSNR: " + " ".join(f"{v:.1f}" for v in ps)
      + f"   min={min(ps):.1f} median={np.median(ps):.1f}", flush=True)
