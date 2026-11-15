# -*- coding: utf-8 -*-
"""渲染：按 camera_path 裁切、缩放、编码，并保留原音轨。"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from core.fastvideo import pipe_frames
from core.video import open_capture
from core.schema import CameraPose


def _encoders() -> str:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True, errors="ignore").stdout
    return out or ""


def pick_encoder(preferred: str) -> str:
    avail = _encoders()
    # hevc_nvenc = H.265 硬编，同画质下比 h264 小 30-50%，代价是兼容性略差
    # （老播放器/AE 旧版本可能不认，所以要 -tag:v hvc1 帮一把）。
    if preferred in ("h264_nvenc", "hevc_nvenc") and preferred in avail:
        return preferred
    return "libx264"


def _enc_args(codec: str, quality: int) -> list[str]:
    if codec == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
                "-cq", str(quality), "-b:v", "0", "-pix_fmt", "yuv420p"]
    if codec == "hevc_nvenc":
        # cq 用同一个质量档：NVENC 的 cq 在 h264/hevc 上尺度接近，参数照抄即可。
        # -tag:v hvc1 让 mp4 里标成 hvc1 而不是默认的 hev1，QuickTime/苹果设备、
        # 以及不少剪辑软件只认 hvc1。
        return ["-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr",
                "-cq", str(quality), "-b:v", "0", "-pix_fmt", "yuv420p",
                "-tag:v", "hvc1"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(quality),
            "-pix_fmt", "yuv420p"]


def _even(v: float) -> int:
    n = int(round(v))
    return n - (n % 2)


def _pipe_size(poses: list) -> tuple[int, int]:
    """管道尺寸取整段最大裁切框。

    变焦素材（源画面自己推拉）每帧需要的裁切尺寸不同，而 ffmpeg 的 rawvideo
    管道尺寸必须固定，所以统一按最大尺寸喂，再在写入前缩放到该尺寸。
    裁切尺寸恒定时（绝大多数素材）这一步完全跳过，没有额外开销。
    """
    pw = max(2, _even(max(float(p.crop_w) for p in poses)))
    ph = max(2, _even(max(float(p.crop_h) for p in poses)))
    return pw, ph


def _grab(frame, pose, pw: int, ph: int) -> np.ndarray:
    """按这一帧自己的裁切框取图，缩放到管道尺寸并补齐越界。"""
    x, y = max(0, int(round(pose.crop_x))), max(0, int(round(pose.crop_y)))
    cw, ch = max(2, int(round(pose.crop_w))), max(2, int(round(pose.crop_h)))
    crop = frame[y:y + ch, x:x + cw]
    if crop.shape[0] != ch or crop.shape[1] != cw or crop.size == 0:
        pad = np.zeros((ch, cw, 3), np.uint8)
        hh, ww = min(crop.shape[0], ch), min(crop.shape[1], cw)
        if hh > 0 and ww > 0:
            pad[:hh, :ww] = crop[:hh, :ww]
        crop = pad
    if crop.shape[1] != pw or crop.shape[0] != ph:
        crop = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_LINEAR)
    return crop


def render_ffmpeg(src: str | Path, poses: list[CameraPose], out: str | Path,
                  fps: float, out_w: int = 1080, out_h: int = 1920,
                  codec: str = "h264_nvenc", quality: int = 19,
                  scale_flags: str = "lanczos", audio: bool = True,
                  limit: int = 0) -> dict:
    """让 ffmpeg 自己逐帧裁剪（sendcmd），Python 完全不下场。

    为什么快：Python 裁剪要先从管道把 25MB/帧 的 4K 数据搬进 numpy，再切片、
    再缩放、再写回管道——实测卡在 32 FPS。sendcmd 版本解码/裁剪/编码在 ffmpeg
    内部流水线并行，实测 **172 FPS**（2583 帧 15 秒），数据一次都不进 Python。

    代价是时间戳对齐只能到"帧"，实测中位 PSNR 36.8dB（个别快速摇镜帧 29dB，
    相当于 1–2px 的位置差）。要像素级精确就还用 Python 路径。
    """
    import time as _t

    src, out = Path(src), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    poses = poses[:limit] if limit else poses
    if not poses:
        raise SystemExit("render_ffmpeg: 没有取景数据")
    cw, ch = _pipe_size(poses)
    sw, sh = 1, 1
    try:
        from core.fastvideo import probe_size
        sw, sh, _ = probe_size(str(src))
    except Exception:                                   # noqa: BLE001
        pass
    cost = _t.time()
    cmd_file = out.parent / (out.stem + ".sendcmd.txt")
    lines = []
    for p in poses:
        # 帧中心时间：用帧边界会差半帧，实测快速摇镜时有 1 帧滞后
        t = (p.fi + 0.02) / max(fps, 1e-6)
        x = max(0, min(int(round(p.crop_x)), max(0, sw - cw)))
        y = max(0, min(int(round(p.crop_y)), max(0, sh - ch)))
        lines.append(f"{t:.6f} crop x {x};")
        lines.append(f"{t:.6f} crop y {y};")
    cmd_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    filt = (f"sendcmd=f='{cmd_file.name}',"
            f"crop={cw}:{ch}:0:0,scale={out_w}:{out_h}:flags={scale_flags}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src.resolve()), "-vf", filt]
    cmd += _enc_args(codec, quality)
    cmd += ["-map", "0:v:0"]
    if audio:
        cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-fps_mode", "passthrough", "-movflags", "+faststart", "-progress", "pipe:1"]
    # 半成品隔离：先写 .partial，成功后再原子改名成 .mp4。
    # 否则渲染到一半被关窗，会留下一个 >0 字节、却只有半段的 mp4，
    # 下一轮"产物齐全"检查会把它当成有效成片误跳过。
    tmp = out.with_name(out.stem + ".partial" + out.suffix)
    cmd += ["-nostats", str(tmp.resolve())]
    t0 = _t.time()
    proc = subprocess.Popen(cmd, cwd=str(out.parent), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    total_t = (len(poses) / max(fps, 1e-6))
    last_pct = -1
    if proc.stdout is not None:
        for line in proc.stdout:
            s = line.decode("utf-8", "ignore").strip()
            if s.startswith("out_time_ms="):
                try:
                    done_t = int(s.split("=", 1)[1]) / 1e6
                except ValueError:
                    continue
                pct = min(99, int(done_t / max(total_t, 1e-6) * 100))
                if pct >= last_pct + 10:        # 每 10% 打一次，别刷屏
                    last_pct = pct
                    el = _t.time() - t0
                    rate = done_t / max(el, 1e-6)
                    eta = max(0, (total_t - done_t) / max(rate, 1e-6))
                    print(f"\r  render {pct:3d}%  {rate:5.1f}x 实时  ETA {eta/60:4.1f} 分钟",
                          end="", flush=True)
    _, err = proc.communicate()
    el = _t.time() - t0
    if proc.returncode != 0:
        err = err.decode("utf-8", "ignore")[-300:]
        cmd_file.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg 内部裁剪失败：{err}")
    tmp.replace(out)                    # 原子改名，半成品隔离的最后一跳
    cmd_file.unlink(missing_ok=True)
    n = len(poses)
    print(f"\r  render {n} 帧  完成 {el:.0f}s（{n / max(el, 1e-6):.1f} FPS，ffmpeg 内部裁剪）"
          + " " * 8)
    return dict(frames=n, seconds=round(el, 1), fps=round(n / max(el, 1e-6), 2),
                codec=codec, out=str(out), size_mb=round(out.stat().st_size / 1e6, 1),
                width=out_w, height=out_h, engine="ffmpeg")


def render(src: str | Path, poses: list[CameraPose], out: str | Path,
           fps: float, out_w: int = 1080, out_h: int = 1920,
           codec: str = "h264_nvenc", quality: int = 19,
           scale_flags: str = "lanczos", audio: bool = True,
           limit: int = 0) -> dict:
    src, out = Path(src), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    codec = pick_encoder(codec)
    poses = poses[:limit] if limit else poses
    cw, ch = _pipe_size(poses)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cw}x{ch}",
           "-r", f"{fps}", "-i", "-"]
    if audio:
        cmd += ["-i", str(src)]
    cmd += ["-vf", f"scale={out_w}:{out_h}:flags={scale_flags}"]
    cmd += _enc_args(codec, quality)
    cmd += ["-map", "0:v:0"]
    if audio:
        cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-movflags", "+faststart"]
    tmp = out.with_name(out.stem + ".partial" + out.suffix)   # 保留 .mp4 扩展名，ffmpeg 才能识别格式
    cmd += [str(tmp)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    t0 = time.time()
    try:
        # ffmpeg 管道读：4K 上 37.7 FPS，OpenCV 只有 22.6 FPS，而且是逐帧零拷贝进来
        for idx, frame in pipe_frames(src, count=len(poses)):
            crop = _grab(frame, poses[idx], cw, ch)
            proc.stdin.write(crop.tobytes())
            n = idx + 1
            if n % 300 == 0:
                el = time.time() - t0
                print(f"\r  render {n}/{len(poses)}  {n/el:5.1f} FPS", end="", flush=True)
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    el = time.time() - t0
    print(f"\r  render {n}/{len(poses)}  {n/max(el,1e-6):5.1f} FPS" + " " * 12)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"ffmpeg 失败（{codec}），检查编码器或输出路径")
    tmp.replace(out)
    return dict(frames=n, seconds=round(el, 1), fps=round(n / max(el, 1e-6), 2),
                codec=codec, out=str(out), size_mb=round(out.stat().st_size / 1e6, 1),
                width=out_w, height=out_h)


def render_multi(src: str | Path, jobs: list[dict], fps: float,
                 out_w: int = 1080, out_h: int = 1920,
                 codec: str = "h264_nvenc", quality: int = 19,
                 scale_flags: str = "lanczos", audio: bool = True,
                 limit: int = 0) -> list[dict]:
    """一次解码、多路输出。

    jobs: [{"out": path, "poses": [CameraPose]}]

    4K/8K 素材上解码本身就是瓶颈（8K 约 70ms/帧）。给 N 个人各渲染一遍等于把
    解码成本乘以 N，所以这里只解一次，把同一帧按各自的取景框裁给 N 个编码器。
    """
    src = Path(src)
    codec = pick_encoder(codec)
    progs: list[subprocess.Popen] = []
    meta = []
    for job in jobs:
        poses = job["poses"]
        if limit:
            poses = poses[:limit]
        job["poses"] = poses
        cw, ch = _pipe_size(poses)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cw}x{ch}",
               "-r", f"{fps}", "-i", "-"]
        if audio:
            cmd += ["-i", str(src)]
        cmd += ["-vf", f"scale={out_w}:{out_h}:flags={scale_flags}"]
        cmd += _enc_args(codec, quality)
        cmd += ["-map", "0:v:0"]
        if audio:
            cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-movflags", "+faststart"]
        tmp = Path(job["out"]).with_name(Path(job["out"]).stem + ".partial" + Path(job["out"]).suffix)
        cmd += [str(tmp)]
        progs.append(subprocess.Popen(cmd, stdin=subprocess.PIPE))
        meta.append(dict(out=str(job["out"]), tmp=tmp, crop=(cw, ch), n=len(poses)))

    n = 0
    t0 = time.time()
    try:
        for n, frame in pipe_frames(src):
            if all(n >= m["n"] for m in meta):
                break
            for job, m, proc in zip(jobs, meta, progs):
                if n >= m["n"]:
                    continue
                pose = job["poses"][n]
                cw, ch = m["crop"]
                crop = _grab(frame, pose, cw, ch)
                try:
                    proc.stdin.write(crop.tobytes())
                except (BrokenPipeError, ValueError):
                    m["n"] = n
            if n % 300 == 0:
                el = time.time() - t0
                print(f"\r  render {n} 帧  {n/el:5.1f} FPS × {len(progs)} 路", end="", flush=True)
    finally:
        for proc in progs:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:      # noqa: BLE001
                    pass
        for proc in progs:
            proc.wait()

    el = time.time() - t0
    print(f"\r  render {n} 帧  {n/max(el,1e-6):5.1f} FPS × {len(progs)} 路" + " " * 10)
    out = []
    for m in meta:
        p = Path(m["out"])
        # 半成品隔离：全部成功才原子改名；失败/中断的 .partial 留着，不算成品
        if m["tmp"].exists() and m["tmp"].stat().st_size > 0:
            m["tmp"].replace(p)
        else:
            m["tmp"].unlink(missing_ok=True)
        out.append(dict(frames=n, seconds=round(el, 1), codec=codec, out=m["out"],
                        size_mb=round(p.stat().st_size / 1e6, 1) if p.exists() else 0.0))
    return out
