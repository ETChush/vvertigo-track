# -*- coding: utf-8 -*-
"""ffmpeg 管道逐帧读取。

为什么不用 cv2.VideoCapture
--------------------------
实测同一个 4K AV1 文件：OpenCV `grab()` 只有 **22.6 FPS**，同样一个 ffmpeg 进程
用管道输出 rawvideo 是 **37.7 FPS**，输出 1280x720 更是 **162 FPS**——OpenCV 的
ffmpeg 后端起进程、做色彩转换、还要多拷一次，链路比裸管道长得多。

两个坑这里都绕开了：
  1. `bufsize=0` 拿到裸 FileIO，`read(n)` **不保证读满 n 字节**。按"读不满就当
     结束"会得到 0 帧的假结果；正确做法是 readinto 到预分配缓冲区。
  2. 直接在 ffmpeg 里 `-vf scale=` 把帧缩到要用的尺寸，比在 Python 里缩 4K
     快得多，也省掉 25MB/帧 的搬运。
"""
from __future__ import annotations

import json
import subprocess
from typing import Iterator

import numpy as np


def probe_size(path: str) -> tuple[int, int, float]:
    """用 ffprobe 拿宽、高、帧率。"""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate",
                        "-of", "json", str(path)], capture_output=True)
    d = json.loads(r.stdout.decode("utf-8", "ignore") or "{}")
    st = (d.get("streams") or [{}])[0]
    num, _, den = str(st.get("r_frame_rate", "30/1")).partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    return int(st.get("width", 0)), int(st.get("height", 0)), fps


def pipe_frames(path: str, *, width: int = 0, height: int = 0,
                count: int = 0, flags: str = "bicubic") -> Iterator[tuple[int, np.ndarray]]:
    """逐帧产出 (index, BGR)。width/height 传 0 表示保持原尺寸。

    返回的数组是复用的缓冲区，需要保留就自己 copy。
    """
    ow, oh, _ = probe_size(path)
    if not (width and height):
        width, height = ow, oh

    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if (width, height) != (ow, oh):
        cmd += ["-vf", f"scale={width}:{height}:flags={flags}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1024 * 1024)
    buf = np.empty((height, width, 3), np.uint8)
    view = buf.reshape(-1)
    nbytes = view.nbytes
    fi = 0
    try:
        while True:
            if p.stdout.readinto(view) != nbytes:
                break
            yield fi, buf
            fi += 1
            if count and fi >= count:
                break
    finally:
        try:
            p.stdout.close()
        except OSError:
            pass
        p.kill()
        p.wait()


def pipe_frames_step(path: str, every: int, *, width: int = 0, height: int = 0,
                     flags: str = "bicubic") -> Iterator[tuple[int, np.ndarray]]:
    """每 every 帧取一帧，在 ffmpeg 侧 select，不把整片搬进 Python。

    4K 上整片走管道是 37.7 FPS（25MB/帧的拷贝卡住），只取每 8 帧后同一件事
    21 秒就能跑完（2583 帧的片子）。yield 的序号是**输出序号**，源帧号 = k * every。
    """
    ow, oh, _ = probe_size(path)
    if not (width and height):
        width, height = ow, oh
    filters = [f"select='not(mod(n\\,{every}))'"]
    if (width, height) != (ow, oh):
        filters.append(f"scale={width}:{height}:flags={flags}")
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-vf", ",".join(filters), "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1024 * 1024)
    buf = np.empty((height, width, 3), np.uint8)
    view = buf.reshape(-1)
    nbytes = view.nbytes
    k = 0
    try:
        while True:
            if p.stdout.readinto(view) != nbytes:
                break
            yield k, buf
            k += 1
    finally:
        try:
            p.stdout.close()
        except OSError:
            pass
        p.kill()
        p.wait()


def read_at(path: str, indices, *, width: int = 0, height: int = 0) -> dict:
    """按帧号取若干帧：顺序解码一次，只留下需要的那些。"""
    want = set(int(i) for i in indices)
    out: dict[int, np.ndarray] = {}
    if not want:
        return out
    last = max(want)
    for fi, f in pipe_frames(path, width=width, height=height, count=last + 1):
        if fi in want:
            out[fi] = f.copy()
    return out
