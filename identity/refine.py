# -*- coding: utf-8 -*-
"""把一条局部轨迹按外观突变切成若干段。

为什么必须做这一步：BoT-SORT 在两人交叉时不只是"断轨"，还会让**同一条轨迹
中途换人**（实测 48.6–56s，人体相似度只有 0.45，而真正的目标在旁边 0.82）。
这种错误在轨迹级别做证据融合是抓不到的——整条轨迹的平均特征会把两个人混成
一个人。切段之后，每一段单独参与身份绑定，错的段自然会被否决。

  python -c "from identity.refine import split; ..."   # 在 pipeline/identify 里调用
"""
from __future__ import annotations

import numpy as np

from identity.manager import LocalTrack


def _mean_vec(vs: list[np.ndarray]) -> np.ndarray | None:
    if not vs:
        return None
    m = np.stack(vs).mean(axis=0)
    return (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)


def _slice_track(t: LocalTrack, i0: int, i1: int, key: tuple[int, int]) -> LocalTrack:
    """按帧索引区间 [i0, i1] 复制一条轨迹（含各项特征的区间聚合）。"""
    frames = t.frames[i0:i1]
    boxes = t.boxes[i0:i1]
    scores = t.scores[i0:i1]
    f0, f1 = frames[0], frames[-1]
    nt = LocalTrack(key=key, frames=list(frames), boxes=list(boxes),
                    scores=list(scores), global_id=-1)

    if t.body2_ts:
        vs = [v for fi, v in t.body2_ts if f0 <= fi <= f1]
        nt.body2 = _mean_vec(vs)
    if t.face_ts:
        vs = [v for fi, v, _ in t.face_ts if f0 <= fi <= f1]
        nt.face = _mean_vec(vs)
        nt.face_n = len(vs)
        if nt.face is not None and vs:
            m = np.stack(vs)
            nt.face_cons = float(np.mean(m @ nt.face))
    if t.low_ts:
        vs = [v for fi, v in t.low_ts if f0 <= fi <= f1]
        nt.low = _mean_vec(vs)
    if t.emb_ts is not None and t.emb_t is not None and len(t.emb_ts) == len(t.emb_t):
        keep = [k for k, fi in enumerate(t.emb_ts) if f0 <= fi <= f1]
        if keep:
            nt.emb_t = t.emb_t[keep]
            nt.emb_ts = t.emb_ts[keep]
            nt.emb = _mean_vec([nt.emb_t[k] for k in range(len(keep))])
    if nt.emb is None:
        nt.emb = t.emb
    if nt.body2 is None:
        nt.body2 = t.body2
    return nt


def split(tracks: dict, min_cos: float = 0.50, min_frames: int = 24,
          max_segments: int = 6) -> tuple[dict, dict]:
    """返回 (切段后的轨迹字典, 统计)。不切段时原样返回。"""
    out: dict[tuple[int, int], LocalTrack] = {}
    stats = dict(input=len(tracks), split=0, added=0)

    for key, t in tracks.items():
        ts = getattr(t, "body2_ts", None) or []
        if len(ts) < 6 or t.n < 2 * min_frames:
            out[key] = t
            continue

        fis = [fi for fi, _ in ts]
        vs = [v for _, v in ts]
        sims = np.array([float(vs[i] @ vs[i + 1]) for i in range(len(vs) - 1)])

        # 找突变点：外观掉下去**并且位置也跳了**才切开。
        # 只凭外观会把"原地转身/低头"当成换人——这批舞者转身极频繁，实测
        # 22 条轨迹被切成 80 段，一个人被拆散，最后覆盖率只有一半。
        cuts = []
        for i, sim in enumerate(sims):
            if sim >= min_cos:
                continue
            b0 = t.box_at(fis[i])
            b1 = t.box_at(fis[i + 1])
            if b0 is None or b1 is None:
                cuts.append(i + 1)
                continue
            h = max(1.0, (b1[3] - b1[1]))
            dx = abs((b1[0] + b1[2]) / 2 - (b0[0] + b0[2]) / 2)
            if dx > 1.2 * h:
                cuts.append(i + 1)
        if not cuts:
            out[key] = t
            continue
        bounds = [0] + cuts + [len(vs)]
        segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        segs = [(a, b) for a, b in segs if b - a >= 2]

        # 过短的片段并入相邻段（外观突变也可能是短暂遮挡/转身，不是换人）
        merged: list[list[int]] = []
        for a, b in segs:
            span = fis[b - 1] - fis[a]
            if merged and span < min_frames:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        if len(merged) > max_segments:
            merged = merged[:max_segments - 1] + [[merged[max_segments - 1][0], merged[-1][1]]]
        if len(merged) < 2:
            out[key] = t
            continue

        # 帧索引区间
        used = 0
        for si, (a, b) in enumerate(merged):
            f0, f1 = fis[a], fis[b - 1]
            idx = [k for k, fi in enumerate(t.frames) if f0 <= fi <= f1]
            if len(idx) < min_frames:
                continue
            nk = (key[0], key[1] * 100 + si) if len(merged) > 1 else key
            out[nk] = _slice_track(t, idx[0], idx[-1] + 1, nk)
            used += 1
        if used >= 2:
            stats["split"] += 1
            stats["added"] += used - 1
        else:
            out[key] = t
    stats["output"] = len(out)
    return out, stats
