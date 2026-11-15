# -*- coding: utf-8 -*-
"""点位导出：把逐帧取景框写成 AE 能直接吃的 CSV。

三条设计原则（决定了文件长什么样）：

1. **坐标一律用源视频像素**。插件可能在 AE 里放原片而不是成片，如果导出成片坐标，
   4K 素材上会差 3.5 倍。源坐标 + 裁切尺寸 + 输出尺寸，任何工程都能还原。

2. **一层数据给全，两种用法任选**：
   - 逐帧行：烘焙用，像素级精确
   - kf=1 标记的稀疏关键帧 + 贝塞尔切线：可编辑用，和成片曲线一致
   两条路的误差在导出时就算出来了，写在文件头 `max_error_px` 里，不靠猜。

3. **切线从最终路径反推**，不是从检测点推。最终路径经过了低通、关键帧重建、
   HOLD 冻结、软限幅——只有它的导数才和成片画面一致。用原始检测点算切线，会在
   冻结段给出"还在动"的手柄，AE 里就会慢慢漂。

坐标系：
  x, y   = 裁切框中心，源视频像素，左上为原点
  scale  = out_w / crop_w（AE 里 Scale = scale * 100）
  时间   = frame / source_fps（真实秒，与成片时间轴一致）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

VERSION = "vvertigo-track v1"

COLUMNS = ("frame,time,x,y,crop_w,crop_h,scale,conf,state,"
           "kf,tix,tiy,tox,toy")


# ---------------------------------------------------------------- 关键帧选择
def choose_keyframes(xs: np.ndarray, ys: np.ndarray, states: list[str],
                     eps: float = 1.5, dt_px: float = 16.0) -> np.ndarray:
    """在**最终路径**上重新做一次 RDP，而不是复用上游的 keyframes。

    path.build 里的 RDP 是在冻结和软限幅之前做的，冻结段的位置会被改写，
    直接拿那批下标导出会对不上画面。这里重算一次，顺手把状态切换点全部钉死。
    """
    from camera.path import rdp

    n = len(xs)
    if n < 3:
        return np.arange(n)

    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    keep[rdp(xs, eps, dt_px)] = True
    if float(np.ptp(ys)) > 0.5:
        keep[rdp(ys, eps, dt_px)] = True

    # 状态切换（TRACKING/PREDICT/HOLD/RECOVER）是真实的运动学拐点，
    # 不能让样条自己猜。前后各钉一帧，保证过渡段的速度形状不丢。
    for i in range(1, n):
        if states[i] != states[i - 1]:
            keep[i] = True
            if i >= 2:
                keep[i - 1] = True
    return np.flatnonzero(keep)


def tangents(xs: np.ndarray, ys: np.ndarray, keys: np.ndarray
             ) -> list[tuple[float, float, float, float]]:
    """每个关键帧的贝塞尔手柄偏移 (tix, tiy, tox, toy)，单位源像素。

    切线取相邻关键帧的割线斜率（px/帧），再按段长换算成手柄长度：
    手柄 = 斜率 × 段长 / 3，这是三次贝塞尔退化成 Hermite 的标准关系。
    入切线方向朝前一段（负），出切线朝后一段（正）——和 AE 的约定一致。
    """
    n = len(xs)
    out = []
    for k in range(len(keys)):
        i = int(keys[k])
        a = int(keys[k - 1]) if k > 0 else max(0, i - 1)
        b = int(keys[k + 1]) if k + 1 < len(keys) else min(n - 1, i + 1)
        span = b - a
        if span <= 0:
            out.append((0.0, 0.0, 0.0, 0.0))
            continue
        mx = (xs[b] - xs[a]) / span
        my = (ys[b] - ys[a]) / span
        dt_out = max(1, int(keys[k + 1]) - i) if k + 1 < len(keys) else 1
        dt_in = max(1, i - int(keys[k - 1])) if k > 0 else 1
        out.append((-mx * dt_in / 3.0, -my * dt_in / 3.0,
                    mx * dt_out / 3.0, my * dt_out / 3.0))
    return out


def reconstruct(vals: np.ndarray, keys: np.ndarray,
                hand_out: list[float], hand_in: list[float]) -> tuple[np.ndarray, float]:
    """用导出的关键帧 + 手柄重建曲线，返回重建结果与最大误差（源像素）。

    这一步是自检：如果 AE 按同样的数据插值，误差就是这里的数字。
    """
    n = len(vals)
    out = np.full(n, vals[0], dtype=np.float64)
    for k in range(len(keys) - 1):
        a, b = int(keys[k]), int(keys[k + 1])
        L = b - a
        if L <= 0:
            continue
        u = np.arange(0, L + 1, dtype=np.float64) / L
        p0 = float(vals[a])
        p1 = p0 + hand_out[k]
        p3 = float(vals[b])
        p2 = p3 + hand_in[k + 1]
        out[a:b + 1] = ((1 - u) ** 3) * p0 + 3 * ((1 - u) ** 2) * u * p1 \
            + 3 * (1 - u) * (u ** 2) * p2 + (u ** 3) * p3
    err = float(np.max(np.abs(out - vals))) if n else 0.0
    return out, err


def _fit(xs: np.ndarray, ys: np.ndarray, keys: np.ndarray,
         tol: float = 1.0, max_iter: int = 40) -> tuple[np.ndarray, float]:
    """迭代补点，直到重建曲线和最终路径的偏差落在 tol 以内。

    为什么需要这一步：RDP 只保证"折线"偏离原路径不超过 eps，但 PCHIP/Hermite
    在折点处的曲率是免费的——拐得急的地方会拱出十几二十像素。补点比调 eps 靠谱：
    eps 调小会让静止段塞满关键帧，补点只往真正超差的地方加。
    """
    keys = np.unique(keys)
    err = 0.0
    for _ in range(max_iter):
        tans = tangents(xs, ys, keys)
        rx, ex = reconstruct(xs, keys, [t[2] for t in tans], [t[0] for t in tans])
        ry, ey = reconstruct(ys, keys, [t[3] for t in tans], [t[1] for t in tans])
        err = max(ex, ey)
        e = np.maximum(np.abs(rx - xs), np.abs(ry - ys))
        bad = np.flatnonzero(e > tol)
        if not bad.size:
            break
        # 每段连续超差区只补最差的那一帧（长区间再补中点），避免关键帧爆炸
        add, run_start, prev = [], int(bad[0]), int(bad[0])
        for i in bad[1:]:
            i = int(i)
            if i != prev + 1:
                add.append(run_start + int(np.argmax(e[run_start:prev + 1])))
                if prev - run_start > 30:
                    add.append((run_start + prev) // 2)
                run_start = i
            prev = i
        add.append(run_start + int(np.argmax(e[run_start:prev + 1])))
        if prev - run_start > 30:
            add.append((run_start + prev) // 2)
        new = np.setdiff1d(np.asarray(add, dtype=np.int64), keys)
        if not new.size:
            break
        keys = np.union1d(keys, new)
    return keys, float(err)


# ---------------------------------------------------------------- 写文件
def write_points(path: str | Path, poses, info: dict, *, label: str = "ID1",
                 gid: int = 0, out_w: int = 1080, out_h: int = 0, eps: float = 3.0,
                 dt_px: float = 16.0, tol_out: float = 1.5,
                 extra: dict | None = None) -> dict:
    """写一份点位 CSV。poses 必须是渲染实际使用的那个 camera_path。

    tol_out 是允许的曲线误差，单位**成片像素**（默认 1px）。按裁切比例换算回源
    像素再拟合，所以 4K/8K 素材上不会为了 1 个源像素塞进成百上千个关键帧。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not poses:
        return dict(frames=0, keyframes=0, max_error_px=0.0, path=str(path))

    n = len(poses)
    xs = np.array([float(p.center_x) for p in poses], dtype=np.float64)
    ys = np.array([float(p.crop_y) + float(p.crop_h) / 2.0 for p in poses], dtype=np.float64)
    states = [str(p.state) for p in poses]
    cw, ch = int(poses[0].crop_w), int(poses[0].crop_h)
    scale = float(out_w) / max(1, cw)

    keys = choose_keyframes(xs, ys, states, eps, dt_px)
    keys, max_err = _fit(xs, ys, keys,
                         tol=max(0.3, float(tol_out) / max(scale, 1e-6)))
    tans = tangents(xs, ys, keys)
    _, err_x = reconstruct(xs, keys, [t[2] for t in tans], [t[0] for t in tans])
    _, err_y = reconstruct(ys, keys, [t[3] for t in tans], [t[1] for t in tans])
    max_err = round(max(err_x, err_y), 3)
    err_out = round(max_err * scale, 3)

    kf_of = {int(i): k for k, i in enumerate(keys)}
    fps = float(info["fps"])
    src_name = Path(str(info.get("source_name") or info["path"])).name

    lines = [
        f"# {VERSION}",
        f"# source={src_name} width={int(info['width'])} height={int(info['height'])}"
        f" fps={fps} frames={n}",
        f"# crop_w={cw} crop_h={ch} out_w={int(out_w)}"
        f" out_h={int(out_h) if out_h else int(round(out_w * ch / max(1, cw)))}",
        f"# scale={round(scale, 6)} coord=center,px,source_top_left # origin",
        f"# target={label} gid={gid}"
        + (f" coverage={extra['coverage']}" if extra and extra.get('coverage') is not None else "")
        + (f" median_h={extra['median_h']}" if extra and extra.get('median_h') is not None else "")
        + (f" face={extra['face_sim']}" if extra and extra.get('face_sim') is not None else ""),
        f"# keyframes={len(keys)} frames_per_keyframe={round(n / max(1, len(keys)), 1)}"
        f" max_error_px={err_out} max_error_src_px={max_err} tol_out={tol_out} eps={eps}",
        f"# columns={COLUMNS}",
    ]

    body = []
    for i, p in enumerate(poses):
        k = kf_of.get(i)
        if k is None:
            ti_x = ti_y = to_x = to_y = 0.0
            kf = 0
        else:
            ti_x, ti_y, to_x, to_y = tans[k]
            kf = 1
        body.append(
            f"{i},{i / fps:.4f},{xs[i]:.1f},{ys[i]:.1f},{cw},{ch},"
            f"{scale:.6f},{float(p.confidence):.3f},{p.state},"
            f"{kf},{ti_x:.2f},{ti_y:.2f},{to_x:.2f},{to_y:.2f}")

    path.write_text("\n".join(lines) + "\n" + COLUMNS + "\n"
                    + "\n".join(body) + "\n", encoding="utf-8")
    return dict(frames=n, keyframes=int(len(keys)), max_error_px=err_out,
                max_error_src_px=max_err, crop_w=cw, crop_h=ch,
                scale=round(scale, 4), path=str(path),
                frames_per_keyframe=round(n / max(1, len(keys)), 1))
