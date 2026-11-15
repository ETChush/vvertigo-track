"""解码吞吐基准：ffmpeg 管道 vs OpenCV。

注意一个坑：subprocess.PIPE + bufsize=0 拿到的是裸 FileIO，read(n) 不保证读满
n 字节，按"读不满就当结束"会得到 0 帧的假结果。这里用 readinto 直接读进预分配
的 numpy 缓冲区，既避免这个坑也省掉一次拷贝。
"""
import subprocess
import time

import cv2
import numpy as np

V = "F:/Projects/Dance/测试视频/3-00.01.56.791-00.02.38.694.mp4"


def pipe_fps(extra, w, h, n, tag):
    cmd = ["ffmpeg", "-v", "error", "-i", V] + extra + \
          ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1024 * 1024)
    frame = np.empty((h, w, 3), np.uint8)
    view = frame.reshape(-1)
    t = time.time()
    got = 0
    while got < n:
        if p.stdout.readinto(view) != view.nbytes:
            break
        got += 1
    dt = time.time() - t
    p.stdout.close()
    p.kill()
    p.wait()
    print(f"  {tag:20s} {got:4d} 帧 {dt:5.2f}s = {got / dt:6.1f} FPS  ({view.nbytes / 1e6:.1f}MB/帧)",
          flush=True)
    return got / max(dt, 1e-6)


print("ffmpeg 管道（readinto 零拷贝）：", flush=True)
pipe_fps([], 3840, 2160, 100, "4K 原始")
pipe_fps(["-vf", "scale=1920:1080"], 1920, 1080, 150, "1920x1080")
pipe_fps(["-vf", "scale=1280:720"], 1280, 720, 200, "1280x720")
pipe_fps(["-vf", "scale=960:540"], 960, 540, 200, "960x540")
pipe_fps(["-vf", "scale=1280:720", "-threads", "4"], 1280, 720, 200, "1280x720 t4")

print("OpenCV：", flush=True)
cap = cv2.VideoCapture(V)
t = time.time()
n = 0
while n < 200:
    if not cap.grab():
        break
    n += 1
print(f"  grab-only          {n:4d} 帧 {time.time() - t:5.2f}s = {n / (time.time() - t):6.1f} FPS",
      flush=True)
cap.release()
