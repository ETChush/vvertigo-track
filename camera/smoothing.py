# -*- coding: utf-8 -*-
"""平滑器与运动学约束。

One Euro 比滑动平均好在它按速度自适应：慢速时强平滑（不抖），快速时弱平滑
（不拖尾）。之后再串一级速度/加速度限幅，保证不会出现"横向甩镜头"。
"""
from __future__ import annotations

import math

import numpy as np


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0, fps: float = 60.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.fps = fps
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float) -> float:
        dt = 1.0 / self.fps
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.fps
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


def one_euro(xs: np.ndarray, fps: float, min_cutoff: float = 1.0,
             beta: float = 0.007, d_cutoff: float = 1.0) -> np.ndarray:
    f = OneEuro(min_cutoff, beta, d_cutoff, fps)
    return np.array([f.filter(float(v)) for v in xs], dtype=np.float64)


def limit_kinematics(xs: np.ndarray, fps: float, max_speed: float,
                     max_accel: float) -> np.ndarray:
    """限幅到速度/加速度上界。max_speed 单位像素/秒，max_accel 像素/秒²。"""
    if len(xs) == 0:
        return xs
    vmax = max_speed / fps
    amax = max_accel / (fps * fps)
    out = xs.astype(np.float64).copy()
    v = 0.0
    for i in range(1, len(out)):
        target_v = out[i] - out[i - 1]
        target_v = max(-vmax * 2, min(vmax * 2, target_v))
        v = v + max(-amax, min(amax, target_v - v))
        v = max(-vmax, min(vmax, v))
        out[i] = out[i - 1] + v
    return out


def reversals(xs: np.ndarray, min_delta: float = 12.0) -> int:
    d = np.diff(xs)
    sign = np.sign(d[np.abs(d) > min_delta])
    if len(sign) < 2:
        return 0
    return int(np.sum(sign[1:] != sign[:-1]))
