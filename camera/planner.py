# -*- coding: utf-8 -*-
"""虚拟摄影机。

逐帧把「目标在哪」变成「裁切框在哪」。所有状态都写进 camera_path.json，
渲染阶段只读这个文件——于是调取景策略不需要重跑任何 AI 模型。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from camera import framing, path, smoothing
from core.schema import CameraPose, IdentityObs


def _fill_and_state(series: list[IdentityObs | None], fps: float,
                    predict_s: float, hold_s: float, lost_s: float,
                    recover_s: float) -> tuple[np.ndarray, list[str], np.ndarray]:
    """缺失帧的处理策略：先按速度外推，超过 predict_s 就冻结。

    冻结而不是"继续追"，是为了满足产品原则：宁可画面停在原地，也不要因为猜错
    方向把镜头甩出去、最后落到另一个人身上。
    """
    n = len(series)
    cx = np.full(n, np.nan)
    conf = np.zeros(n)
    for i, s in enumerate(series):
        if s is not None:
            cx[i] = s.cx
            conf[i] = max(0.05, min(1.0, s.identity_conf))

    states = ["TRACKING"] * n
    pred_max = max(1, int(predict_s * fps))
    lost_max = max(pred_max, int(lost_s * fps))

    first = next((i for i in range(n) if not np.isnan(cx[i])), None)
    if first is None:
        return np.full(n, 0.0), ["LOST"] * n, conf
    cx[:first] = cx[first]
    states[:first] = ["HOLD"] * first

    last = first
    for i in range(first + 1, n):
        if not np.isnan(cx[i]):
            last = i
            continue
        gap = i - last
        k0 = max(0, last - 8)
        valid = cx[k0:last + 1]
        valid = valid[~np.isnan(valid)]
        vel = 0.0
        if len(valid) >= 3:
            vel = float(np.polyfit(np.arange(len(valid)), valid, 1)[0])
            vel = float(np.clip(vel, -25.0, 25.0))
        if gap <= pred_max:
            cx[i] = cx[last] + vel * gap
            states[i] = "PREDICT"
        else:
            cx[i] = cx[last] + vel * pred_max
            states[i] = "HOLD" if gap <= lost_max else "LOST"

    # 重新捕获：空档 >= 0.25s 之后的第一段，标记 RECOVER，提醒"这是过渡，不是稳态"
    gaps = np.isnan(np.array([1.0 if s is None else np.nan for s in series], dtype=np.float64))
    rec = int(recover_s * fps)
    min_gap = max(1, int(0.25 * fps))
    i = 0
    while i < n:
        if series[i] is not None:
            j = i
            while j < n and series[j] is not None:
                j += 1
            if i > 0:
                prev_end = i - 1
                k = prev_end
                while k >= 0 and series[k] is None:
                    k -= 1
                if prev_end - k >= min_gap:
                    for m in range(i, min(j, i + rec)):
                        states[m] = "RECOVER"
            i = j
        else:
            i += 1
    _ = gaps
    return cx, states, conf


def plan(series: list[IdentityObs | None], info: dict, ccfg: dict,
         target_id: int) -> tuple[list[CameraPose], dict]:
    n = len(series)
    fps = float(info["fps"])
    W, H = int(info["width"]), int(info["height"])

    cw, ch = framing.crop_size(W, H, ccfg.get("aspect", "9:16"),
                               bool(ccfg.get("use_full_height", True)),
                               int(ccfg.get("crop_width") or 0),
                               float(ccfg.get("zoom", 1.0)))
    anchored_y = 0 if ch >= H else int(round((H - ch) * float(ccfg.get("anchor_y", 0.5))))

    cx, states, conf = _fill_and_state(
        series, fps,
        float(ccfg.get("predict_seconds", 1.2)),
        float(ccfg.get("hold_after_seconds", 1.2)),
        float(ccfg.get("lost_after_seconds", 8.0)),
        float(ccfg.get("recover_seconds", 0.8)))

    # 路径生成：默认走"关键帧 + 样条"（AE 式），也保留旧的 One Euro 做对照
    mode = str(ccfg.get("path_mode", "keyframe")).lower()
    if mode == "keyframe":
        sm, path_info = path.build(cx, states, fps, ccfg)
    else:
        oe = ccfg.get("one_euro", {}) or {}
        sm = smoothing.one_euro(cx, fps,
                                float(oe.get("min_cutoff", 1.1)),
                                float(oe.get("beta", 0.008)),
                                float(oe.get("d_cutoff", 1.0)))
        sm = smoothing.limit_kinematics(sm, fps,
                                        float(ccfg.get("max_speed", 900)),
                                        float(ccfg.get("max_accel", 3200)))
        path_info = {}

    margin = int(ccfg.get("edge_margin", 0))

    # 纵向路径：只有 zoom>1（裁切高度小于源高）时才需要跟随
    vcfg = ccfg.get("vertical") or {}
    if ch >= H:
        cy_path = np.full(n, H / 2.0)
        v_info = dict(follow=False)
    else:
        headroom = float(vcfg.get("headroom", 0.06))
        cy_raw = np.full(n, np.nan)
        for i, s in enumerate(series):
            if s is not None:
                cy_raw[i] = framing.crop_y_center(s.y1, s.y2, ch, H, headroom)
        ok = np.flatnonzero(~np.isnan(cy_raw))
        if len(ok):
            cy_raw[: ok[0]] = cy_raw[ok[0]]
            cy_raw[ok[-1] + 1:] = cy_raw[ok[-1]]
            for i in range(n):
                if np.isnan(cy_raw[i]):
                    cy_raw[i] = cy_raw[i - 1]
        else:
            cy_raw[:] = H / 2.0
        vsub = dict(ccfg)
        vsub.update(smooth_sigma=float(vcfg.get("smooth_sigma", 26)),
                    simplify_eps=float(vcfg.get("simplify_eps", 4.0)),
                    preset="custom")
        cy_path, v_info = path.build(cy_raw, states, fps, vsub)
        v_info["follow"] = True
        v_info["headroom"] = headroom

    poses: list[CameraPose] = []
    for i in range(n):
        x = framing.crop_x_for(float(sm[i]), cw, W, margin)
        y = anchored_y if ch >= H else framing.crop_y_for(float(cy_path[i]), ch, H)
        poses.append(CameraPose(fi=i, t=i / fps, crop_x=x, crop_y=y,
                                crop_w=cw, crop_h=ch, center_x=float(sm[i]),
                                target_id=target_id, state=states[i],
                                confidence=float(conf[i])))

    raw = np.array([s.cx if s is not None else np.nan for s in series], dtype=np.float64)
    inside = 0
    head_ok = 0
    foot_ok = 0
    for i, s in enumerate(series):
        if s is None:
            continue
        p = poses[i]
        if p.crop_x <= s.x1 and s.x2 <= p.crop_x + cw:
            inside += 1
        if s.y1 >= p.crop_y:
            head_ok += 1
        if s.y2 <= p.crop_y + ch:
            foot_ok += 1
    n_vis = int(np.sum(~np.isnan(raw)))
    v = np.diff(sm)
    summary = dict(
        target_id=target_id, crop_w=cw, crop_h=ch,
        crop_y=anchored_y if ch >= H else -1,   # -1 = 逐帧跟随，见 vertical
        upscale=round(framing.upscale_factor(cw, int(ccfg.get("out_width", 1080))), 3),
        frames=n, target_visible=n_vis, visible_ratio=round(n_vis / max(1, n), 4),
        target_inside_crop=inside,
        target_inside_ratio=round(inside / max(1, n_vis), 4),
        head_inside_ratio=round(head_ok / max(1, n_vis), 4),
        foot_inside_ratio=round(foot_ok / max(1, n_vis), 4),
        states={s: states.count(s) for s in sorted(set(states))},
        mean_speed_px_s=round(float(np.mean(np.abs(v)) * fps), 1) if len(v) else 0.0,
        max_speed_px_s=round(float(np.max(np.abs(v)) * fps), 1) if len(v) else 0.0,
        reversals=smoothing.reversals(sm, 12.0),
        mean_offset_px=round(float(np.mean(np.abs(
            [sm[i] - (poses[i].crop_x + cw / 2) for i in range(n)]))), 1),
        path=path_info, vertical=v_info,
    )
    return poses, summary
