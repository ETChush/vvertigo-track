# -*- coding: utf-8 -*-
"""取景路径：全帧点位 → 关键帧删减 → 样条重建。

对应 AE 里手工跟踪的工作流，只是自动完成：

  1. 拿到全帧点位（每帧一个观测）
  2. 零相位低通，压掉检测框的手臂摆动噪声
  3. Douglas-Peucker 删减关键帧——**位置自适应**，不是固定每 N 帧
  4. PCHIP 样条穿过关键帧（保形、不过冲、C1 连续）

为什么不用固定间隔的关键帧：舞者的真实运动速度差 10 倍以上（慢走位 vs 快速横穿），
固定 5 帧一个会在慢速段塞满无用关键帧、在快速段又不够。RDP 按"偏离直线多少像素"
决定保留，快速段自动加密、静止段自动合并成一个。

为什么不用因果滤波器：因果 IIR 只能靠滞后换平滑，且级联后会互相激励（实测把抖动
从 16.9px 放大到 27.5px）。零相位（正反各滤一次）没有滞后，可以用更强的 cutoff。
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

HOLD_STATES = ("HOLD", "LOST")

# 三档预设。数值来自 seg_002 上的实测（见 tools/path_compare.py）：
#   follow 跟得紧，快速横穿时最不容易出框，但保留一点手感的颗粒
#   smooth 默认，兼顾跟随与稳定
#   steady 最稳，接近三脚架上慢慢摇的效果，代价是快速走位时略慢半拍
PRESETS = {
    "follow": dict(smooth_sigma=9.0, simplify_eps=1.5, time_weight=16.0),
    "smooth": dict(smooth_sigma=16.0, simplify_eps=3.0, time_weight=16.0),
    "steady": dict(smooth_sigma=22.0, simplify_eps=5.0, time_weight=20.0),
}


def rdp(x: np.ndarray, eps: float, dt_px: float = 16.0) -> np.ndarray:
    """Douglas-Peucker 简化，返回保留的关键帧下标。

    坐标用 (帧号×dt_px, 位置)，dt_px 让"时间"和"像素"可比。eps 单位是像素——含义
    很直白：允许样条偏离原始点位多少。
    """
    n = len(x)
    if n < 3:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        ta, tb = a * dt_px, b * dt_px
        xa, xb = x[a], x[b]
        seg = np.hypot(tb - ta, xb - xa)
        idx = np.arange(a + 1, b)
        if seg < 1e-9:
            d = np.abs(x[a + 1:b] - xa)
        else:
            d = np.abs((xb - xa) * (idx * dt_px - ta)
                       - (tb - ta) * (x[a + 1:b] - xa)) / seg
        k = int(np.argmax(d))
        if d[k] > eps:
            m = a + 1 + k
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return np.flatnonzero(keep)


def rebuild(keys: np.ndarray, x: np.ndarray) -> np.ndarray:
    """PCHIP 穿过关键帧。保形意味着不会在关键帧之间冲过头——这正是贝塞尔手柄要防的。"""
    t = np.arange(len(x), dtype=np.float64)
    if len(keys) < 2:
        return np.full(len(x), x[keys[0]] if len(keys) else 0.0)
    return PchipInterpolator(t[keys], x[keys], extrapolate=True)(t)


def freeze(y: np.ndarray, states, ramp: int = 15) -> np.ndarray:
    """HOLD 段整体冻结，并在之后短暂过渡回跟踪，避免边界出现速度突变。

    产品原则：目标不可见时宁可停住，也不猜方向把镜头甩出去。
    """
    out = y.copy()
    st = np.array(states)
    mask = np.isin(st, HOLD_STATES)
    n = len(out)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        out[i:j] = out[i]                       # 整段锁在进入 HOLD 时的位置
        if j < n and ramp > 1:
            k = min(n, j + ramp)
            # 从冻结值平滑过渡回重建路径，smoothstep 保证两端速度为 0
            w = np.linspace(0.0, 1.0, k - j)
            w = w * w * (3 - 2 * w)
            out[j:k] = out[j - 1] * (1 - w) + y[j:k] * w
        i = j
    return out


def soft_limit(x: np.ndarray, fps: float, max_speed: float, max_accel: float,
               passes: int = 6, win: int = 3) -> np.ndarray:
    """非因果软限幅：只对超限的局部做邻域平均，不整体重排速度。

    平滑后的路径通常远低于阈值（实测 |a|p95 ≈ 340，阈值 3200），这里只是安全网。
    刻意不做"由加速度积分重建速度"——那会引入整体漂移，因果限幅器正是这么把
    抖动放大的。
    """
    y = np.asarray(x, dtype=np.float64).copy()
    if len(y) < 5:
        return y
    vmax = max_speed / fps
    amax = max_accel / (fps * fps)
    for _ in range(passes):
        v = np.diff(y)
        a = np.diff(y, 2)
        bad = np.zeros(len(y), dtype=bool)
        for i in np.flatnonzero(np.abs(v) > vmax):
            bad[max(0, i - 1):i + 2] = True
        for i in np.flatnonzero(np.abs(a) > amax):
            bad[max(0, i - 1):i + 3] = True
        if not bad.any():
            break
        idx = np.flatnonzero(bad)
        y2 = y.copy()
        for i in idx:
            lo, hi = max(0, i - win), min(len(y), i + win + 1)
            y2[i] = y[lo:hi].mean()
        y = y2
    return y


def build(cx: np.ndarray, states, fps: float, cfg: dict) -> tuple[np.ndarray, dict]:
    """完整路径：低通 → 关键帧 → 样条 → 冻结 → 软限幅。"""
    preset = str(cfg.get("preset", "smooth")).lower()
    base = dict(PRESETS.get(preset, PRESETS["smooth"]))
    # 显式写在配置里的值优先于预设（方便只改一项做对照）
    for k in ("smooth_sigma", "simplify_eps", "time_weight"):
        if cfg.get(k) is not None:
            base[k] = float(cfg[k])
    sigma = float(base["smooth_sigma"])
    eps = float(base["simplify_eps"])
    dt_px = float(base["time_weight"])
    ramp = int(round(float(cfg.get("hold_ramp_s", 0.25)) * fps))

    x = np.asarray(cx, dtype=np.float64)
    pre = gaussian_filter1d(x, max(1e-3, sigma), mode="nearest") if sigma > 0 else x.copy()
    keys = rdp(pre, eps, dt_px) if eps > 0 else np.arange(len(x))
    path = rebuild(keys, pre)
    path = freeze(path, states, ramp)

    if cfg.get("soft_limit", True):
        path = soft_limit(path, fps,
                          float(cfg.get("max_speed", 900)),
                          float(cfg.get("max_accel", 3200)))

    v = np.diff(path) * fps
    a = np.diff(path, 2) * fps * fps
    j = np.diff(path, 3) * fps ** 3
    info = dict(
        keyframes=int(len(keys)),
        keyframes_per_sec=round(len(keys) / max(1e-6, len(x) / fps), 2),
        frames_per_keyframe=round(len(x) / max(1, len(keys)), 1),
        preset=preset, smooth_sigma=sigma, simplify_eps=eps,
        max_speed_px_s=round(float(np.max(np.abs(v))), 1) if len(v) else 0.0,
        accel_p95=round(float(np.percentile(np.abs(a), 95)), 1) if len(a) else 0.0,
        jerk_p50=round(float(np.percentile(np.abs(j), 50)), 1) if len(j) else 0.0,
    )
    return path, info
