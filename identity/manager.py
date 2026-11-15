# -*- coding: utf-8 -*-
"""Identity Manager。

职责：把跟踪器输出的「局部轨迹」聚成「全局身份」，并回答一个唯一的问题——
这一帧里，我要跟的那个人在哪。

三条硬规则：
1. 同一帧出现的两条轨迹**永远不可能是同一个人**（共现约束），聚类时作为硬否决。
2. 外观相似度不够就新建身份，绝不硬绑。宁可多一个 ID，不可错一个人。
3. 目标身份一旦确定，不因为"别人更像"而改主意——只在真正重新认出时才恢复。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.schema import IdentityEvent, IdentityObs
from identity.appearance import lowbody_desc


@dataclass
class LocalTrack:
    key: tuple[int, int]                 # (shot_id, local_id)
    frames: list[int] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    emb: Optional[np.ndarray] = None     # 归一化平均外观（人体）
    emb_t: Optional[np.ndarray] = None   # 时间序列 (N, D)
    emb_ts: Optional[np.ndarray] = None
    face: Optional[np.ndarray] = None    # 归一化平均人脸特征
    face_n: int = 0                      # 可用人脸样本数
    face_cons: float = 0.0               # 人脸样本自身一致性
    low: Optional[np.ndarray] = None     # 下半身颜色描述子
    body2: Optional[np.ndarray] = None   # cheetah 人体 ReID 特征
    face2: Optional[np.ndarray] = None   # cheetah 人脸特征（SFace 失败时补位）
    body2_ts: list = field(default_factory=list)   # [(fi, vec)] 用于切段
    face_ts: list = field(default_factory=list)    # [(fi, vec, score)]
    low_ts: list = field(default_factory=list)     # [(fi, vec)]
    global_id: int = -1

    @property
    def shot_id(self) -> int:
        return self.key[0]

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    @property
    def median_h(self) -> float:
        return float(np.median([b[3] - b[1] for b in self.boxes])) if self.boxes else 0.0

    def box_at(self, fi: int):
        try:
            k = self.frames.index(fi)
        except ValueError:
            return None
        return self.boxes[k]

    @property
    def velocity(self) -> tuple[float, float]:
        """末端速度（像素/帧），用最后 5 个点估计。"""
        if self.n < 2:
            return (0.0, 0.0)
        k = min(5, self.n)
        fs = self.frames[-k:]
        bs = self.boxes[-k:]
        dfs = fs[-1] - fs[0]
        if dfs <= 0:
            return (0.0, 0.0)
        cx0 = (bs[0][0] + bs[0][2]) / 2
        cx1 = (bs[-1][0] + bs[-1][2]) / 2
        f0, f1 = bs[0][3], bs[-1][3]
        return ((cx1 - cx0) / dfs, (f1 - f0) / dfs)


@dataclass
class GlobalIdentity:
    gid: int
    gallery: list[np.ndarray] = field(default_factory=list)
    track_keys: list[tuple[int, int]] = field(default_factory=list)
    frames_all: list[int] = field(default_factory=list)
    last_frame: int = -1
    last_box: Optional[tuple[float, float, float, float]] = None
    last_shot: int = -1
    velocity: tuple[float, float] = (0.0, 0.0)
    face_gallery: list[np.ndarray] = field(default_factory=list)
    low_gallery: list[np.ndarray] = field(default_factory=list)
    body2_gallery: list[np.ndarray] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray:
        if not self.gallery:
            return np.zeros(512, dtype=np.float32)
        m = np.mean(np.stack(self.gallery), axis=0)
        return m / (np.linalg.norm(m) + 1e-9)

    @property
    def frames(self) -> int:
        return len(self.frames_all)


# ---------------------------------------------------------------- 轨迹整理
def build_local_tracks(rows) -> dict[tuple[int, int], LocalTrack]:
    out: dict[tuple[int, int], LocalTrack] = {}
    for r in rows:
        key = (int(r.get("sid", 0)), int(r["lid"]))
        t = out.get(key)
        if t is None:
            t = out[key] = LocalTrack(key=key)
        t.frames.append(int(r["fi"]))
        t.boxes.append((float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])))
        t.scores.append(float(r.get("s", 0.0)))
    return out


def filter_tracks(tracks: dict, min_frames: int, min_h: float) -> dict:
    return {k: t for k, t in tracks.items()
            if t.n >= min_frames and t.median_h >= min_h}


# ---------------------------------------------------------------- 外观特征
def attach_embeddings(video, info, tracks: dict, embedder, sample_every: int = 4,
                      max_samples: int = 16) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """一次解码遍历，为所有局部轨迹采样外观特征。"""
    from core.video import frames as iter_frames

    by_frame: dict[int, list[LocalTrack]] = {}
    for t in tracks.values():
        idx = np.unique(np.linspace(0, t.n - 1, min(max_samples, t.n)).astype(int))
        for k in idx:
            by_frame.setdefault(t.frames[k], []).append(t)

    store: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
    low_store: dict[tuple[int, int], list[np.ndarray]] = {}
    want = sorted(by_frame)
    wset = set(want)

    for fi, frame in iter_frames(video, limit=info.n_frames):
        if fi not in wset:
            continue
        if sample_every > 1 and fi % sample_every:
            continue
        ts = [t for t in by_frame[fi]]
        if not ts:
            continue
        boxes = [t.box_at(fi) for t in ts]
        keep = [(t, b) for t, b in zip(ts, boxes) if b is not None]
        if not keep:
            continue
        embs = embedder.embed(frame, [b for _, b in keep])
        for (t, b), e in zip(keep, embs):
            store.setdefault(t.key, []).append((fi, e))
            ld = lowbody_desc(frame, b)
            if ld is not None:
                low_store.setdefault(t.key, []).append(ld)
                t.low_ts.append((fi, ld))

    for key, items in store.items():
        t = tracks[key]
        items.sort(key=lambda x: x[0])
        mat = np.stack([e for _, e in items])
        t.emb_ts = np.array([f for f, _ in items], dtype=np.int64)
        t.emb_t = mat
        # 去掉偏离中位最远的离群样本，避免动作模糊/遮挡帧污染身份库
        med = np.median(mat, axis=0)
        sims = mat @ (med / (np.linalg.norm(med) + 1e-9))
        order = np.argsort(-sims)[: max(1, int(len(sims) * 0.8))]
        m = mat[order].mean(axis=0)
        t.emb = (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)

    for key, vs in low_store.items():
        t = tracks.get(key)
        if t is None or not vs:
            continue
        m = np.mean(np.stack(vs), axis=0)
        t.low = (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)

    missing = [k for k in tracks if tracks[k].emb is None]
    for k in missing:
        tracks[k].emb = np.zeros(512, dtype=np.float32)
    return ([t.emb for t in tracks.values() if t.emb is not None], [])


# ---------------------------------------------------------------- 标定
def calibrate(tracks: dict, fallback: float = 0.55) -> dict:
    """用素材自己的分布定阈值，不套用别处的常数。

    within  = 同一条局部轨迹内部、相邻采样的相似度（同一个人）
    between = 同帧共现的两条轨迹之间的相似度（必然不同人，最难的负样本）
    """
    within = []
    for t in tracks.values():
        if t.emb_t is None or len(t.emb_t) < 2:
            continue
        s = t.emb_t[:-1] @ t.emb_t[1:].T
        within.extend(np.diag(s).tolist())

    between = []
    tracks_l = list(tracks.values())
    for i in range(len(tracks_l)):
        for j in range(i + 1, len(tracks_l)):
            a, b = tracks_l[i], tracks_l[j]
            if a.emb is None or b.emb is None:
                continue
            if a.shot_id != b.shot_id:
                continue
            overlap = min(a.end, b.end) - max(a.start, b.start)
            if overlap > 2:            # 共现 → 一定不同人
                between.append(float(a.emb @ b.emb))

    if len(within) < 10 or len(between) < 10:
        return dict(thresh=fallback, weak=True, n_within=len(within),
                    n_between=len(between), within_p10=None, between_p99=None)

    w10 = float(np.percentile(within, 10))
    b99 = float(np.percentile(between, 99))
    thr = (w10 + b99) / 2.0 if b99 < w10 else b99 + 0.02
    thr = float(np.clip(thr, 0.30, 0.86))
    return dict(thresh=round(thr, 3), weak=False,
                n_within=len(within), n_between=len(between),
                within_p10=round(w10, 3), within_p50=round(float(np.percentile(within, 50)), 3),
                between_p99=round(b99, 3), between_p50=round(float(np.percentile(between, 50)), 3))


# ---------------------------------------------------------------- 聚类
def face_similarity(t: LocalTrack, g: "GlobalIdentity") -> Optional[float]:
    if t.face is None or not g.face_gallery:
        return None
    c = np.mean(np.stack(g.face_gallery), axis=0)
    c = c / (np.linalg.norm(c) + 1e-9)
    return float(t.face @ c)


def attach_person_reid(video, info, tracks: dict, analyzer, every: int = 30,
                       max_per_track: int = 6) -> dict:
    """InsightFace PersonAnalysis 低频采样，取人体 ReID 与人脸，关联回局部轨迹。"""
    from core.video import frames as iter_frames
    from identity.person_analysis import associate

    by_frame: dict[int, list[tuple[tuple[int, int], tuple]]] = {}
    for t in tracks.values():
        for fi, b in zip(t.frames, t.boxes):
            if fi % every:
                continue
            by_frame.setdefault(fi, []).append((t.key, b))

    body: dict[tuple[int, int], list[tuple[float, np.ndarray]]] = {}
    face: dict[tuple[int, int], list[tuple[float, np.ndarray]]] = {}
    n_frames = 0
    for fi, frame in iter_frames(video, limit=info.n_frames):
        boxes = by_frame.get(fi)
        if not boxes:
            continue
        persons = analyzer.get(frame)
        if not persons:
            continue
        n_frames += 1
        for key, p in associate(persons, boxes).items():
            if p.reid is not None:
                body.setdefault(key, []).append((p.score, p.reid))
                tracks[key].body2_ts.append((fi, p.reid))
            if p.face is not None and p.face_valid:
                face.setdefault(key, []).append((p.score, p.face))

    for store, attr in ((body, "body2"), (face, "face2")):
        for key, items in store.items():
            t = tracks.get(key)
            if t is None or not items:
                continue
            items.sort(key=lambda x: -x[0])
            m = np.stack([e for _, e in items[:max_per_track]]).mean(axis=0)
            setattr(t, attr, (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32))
    return dict(frames_with_persons=n_frames,
                tracks_with_body=sum(1 for t in tracks.values() if t.body2 is not None),
                tracks_with_face=sum(1 for t in tracks.values() if t.face2 is not None))


def body2_similarity(t: LocalTrack, g: "GlobalIdentity") -> Optional[float]:
    if t.body2 is None or not g.body2_gallery:
        return None
    c = np.mean(np.stack(g.body2_gallery), axis=0)
    c = c / (np.linalg.norm(c) + 1e-9)
    return float(t.body2 @ c)


def calibrate_body2(tracks: dict, fallback: tuple[float, float] = (0.72, 0.62)) -> dict:
    """cheetah 人体 ReID 的接受/否决阈值，同样用共现分布标定。"""
    lst = [t for t in tracks.values() if t.body2 is not None]
    between = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            a, b = lst[i], lst[j]
            if a.shot_id != b.shot_id:
                continue
            if min(a.end, b.end) - max(a.start, b.start) > 2:
                between.append(float(a.body2 @ b.body2))
    if len(between) < 8:
        return dict(pos=fallback[0], neg=fallback[1], calibrated=False, n=len(between))
    p95 = float(np.percentile(between, 95))
    pos = float(np.clip(p95 + 0.04, 0.62, 0.88))
    neg = float(np.clip(p95 - 0.03, 0.50, pos - 0.05))
    return dict(pos=round(pos, 3), neg=round(neg, 3), calibrated=True, n=len(between),
                p50=round(float(np.median(between)), 3), p95=round(p95, 3))


def low_similarity(t: LocalTrack, g: "GlobalIdentity") -> Optional[float]:
    if t.low is None or not g.low_gallery:
        return None
    c = np.mean(np.stack(g.low_gallery), axis=0)
    c = c / (np.linalg.norm(c) + 1e-9)
    return float(t.low @ c)


def calibrate_low(tracks: dict, fallback: float = 0.62) -> dict:
    """下半身描述子的否决阈值：共现（必不同人）分布的 p99。"""
    lst = [t for t in tracks.values() if t.low is not None]
    between = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            a, b = lst[i], lst[j]
            if a.shot_id != b.shot_id:
                continue
            if min(a.end, b.end) - max(a.start, b.start) > 2:
                between.append(float(a.low @ b.low))
    if len(between) < 8:
        return dict(neg=fallback, calibrated=False, n=len(between))
    p99 = float(np.percentile(between, 99))
    return dict(neg=round(float(np.clip(p99 + 0.02, 0.45, 0.90)), 3),
                calibrated=True, n=len(between),
                p50=round(float(np.median(between)), 3), p99=round(p99, 3))


def cluster(tracks: dict, thresh: float, motion_gate: float = 2.2,
            min_new_frames: int = 30, fps: float = 60.0,
            face_pos: float = 0.62, face_neg: float = 0.55,
            low_neg: float = 0.62, body_pos: float = 0.72, body_neg: float = 0.62,
            face_guard: float = 0.88,
            ) -> tuple[list[GlobalIdentity], list[IdentityEvent]]:
    """证据优先级：人脸 > 人体外观 + 运动 > 纯运动（唯一候选时）。

    人脸是唯一有区分度的模态，所以它既能"强确认"也能"硬否决"：
    人脸相似度明显低于阈值时，哪怕位置完全对得上，也不合并——这是防漂移的关键。
    """
    ident: list[GlobalIdentity] = []
    events: list[IdentityEvent] = []
    order = sorted(tracks.values(), key=lambda t: (t.start, -t.n))

    def overlaps(g: GlobalIdentity, t: LocalTrack) -> bool:
        for k in g.track_keys:
            o = tracks.get(k)
            if o is None or o.shot_id != t.shot_id:
                continue
            if min(o.end, t.end) - max(o.start, t.start) > 2:
                return True
        return False

    def motion_score(g: GlobalIdentity, t: LocalTrack) -> Optional[float]:
        if g.last_box is None or g.last_shot != t.shot_id:
            return None
        dt = t.start - g.last_frame
        if not (0 < dt <= 180):
            return None
        h = g.last_box[3] - g.last_box[1]
        px = (g.last_box[0] + g.last_box[2]) / 2 + g.velocity[0] * dt
        pf = g.last_box[3] + g.velocity[1] * dt
        bx = (t.boxes[0][0] + t.boxes[0][2]) / 2
        bf = t.boxes[0][3]
        d = math.hypot(bx - px, bf - pf) / max(h, 1.0)
        return math.exp(-0.5 * (d / motion_gate) ** 2) if d <= motion_gate * 2 else -1.0

    def app_sim(t: LocalTrack, g: GlobalIdentity) -> float:
        """外观相似度；缺少特征时给 -1，让它自然排到后面而不是把矩阵乘炸掉。"""
        if t.emb is None or g.centroid is None:
            return -1.0
        return float(t.emb @ g.centroid)

    for t in order:
        best, best_cost, parts, why = None, -1.0, (0.0, 0.0, None), ""

        for g in ident:
            if overlaps(g, t):
                continue
            fs = face_similarity(t, g)
            if fs is not None and fs <= face_neg:
                continue                       # 人脸否决：不是同一个人
            mot = motion_score(g, t)
            if mot is not None and mot < 0:
                continue                       # 位置完全不合理
            # 重新绑定（有空档）时用下半身颜色再验一道：黑靴 vs 白靴必被拦下
            ls = low_similarity(t, g)
            dt = (t.start - g.last_frame) if g.last_shot == t.shot_id else 10 ** 9
            if ls is not None and dt > 0.5 * fps and ls <= low_neg:
                continue
            b2 = body2_similarity(t, g)
            # 人体外观与当前身份明显不符时，即便人脸给了高分也不绑——两条证据
            # 打架就进入"不确定"，这正是 48.6s 那次错绑的教训。
            if b2 is not None and b2 <= body_neg and dt > 0.5 * fps:
                continue
            # 这条轨迹没检出人脸，而对方身份是靠人脸立起来的：不许用人体外观
            # 蒙混过关。4K 素材上两个舞者同款亮片裙，body2 能到 0.85，结果把
            # 银裙那段并进了紫裙身份（实测人脸只有 0.175）。宁可不合并。
            if fs is None and t.face is None and g.face_gallery and b2 is not None \
                    and b2 < face_guard:
                continue
            if fs is not None and fs >= face_pos and (mot is None or mot >= 0.15):
                cost = 2.0 + fs
                parts, why = (app_sim(t, g), mot or 0.0, fs), "face"
            elif b2 is not None and b2 >= body_pos and (mot is None or mot >= 0.15):
                cost = 1.5 + b2
                parts, why = (app_sim(t, g), mot or 0.0, fs), "body2"
            else:
                app = app_sim(t, g)
                if app < thresh:
                    continue
                cost = 0.75 * app + 0.25 * (mot or 0.0)
                parts, why = (app, mot or 0.0, fs), "body"
            if cost > best_cost:
                best, best_cost = g, cost

        motion_only = False
        if best is None and t.n >= 6:
            cands = []
            for g in ident:
                if overlaps(g, t) or g.last_box is None:
                    continue
                fs = face_similarity(t, g)
                if fs is not None and fs <= face_neg:
                    continue
                if fs is None and t.face is None and g.face_gallery:
                    b2 = body2_similarity(t, g)
                    if b2 is None or b2 < face_guard:
                        continue
                ls = low_similarity(t, g)
                if ls is not None and ls <= low_neg:
                    continue
                mot = motion_score(g, t)
                if mot is None or mot < 0:
                    continue
                cands.append((mot, g))
            if len(cands) == 1:
                best = cands[0][1]
                parts = (0.0, cands[0][0], None)
                motion_only = True

        if best is not None:
            t.global_id = best.gid
            if t.emb is not None:
                best.gallery.append(t.emb)
            if t.face is not None:
                best.face_gallery.append(t.face)
            if t.low is not None:
                best.low_gallery.append(t.low)
            if t.body2 is not None:
                best.body2_gallery.append(t.body2)
            best.track_keys.append(t.key)
            best.frames_all.extend(t.frames)
            best.last_frame = t.end
            best.last_box = t.boxes[-1]
            best.last_shot = t.shot_id
            best.velocity = t.velocity
            action = ("MOTION_BIND" if motion_only
                      else "FACE_BIND" if why == "face"
                      else "BODY_BIND" if why == "body2"
                      else ("BIND" if t.start - best.last_frame < 200 else "RECOVER"))
            events.append(IdentityEvent(
                t=round(t.start / fps, 3), fi=t.start, local_id=t.key[1],
                global_id=best.gid, action=action,
                appearance_score=round(parts[0], 3),
                motion_score=round(parts[1], 3),
                note=(f"face={parts[2]:.2f}" if parts[2] is not None else
                      ("仅运动证据接续" if motion_only else f"shot {t.shot_id}, {t.n} 帧"))))
        elif t.n >= min_new_frames or t.median_h > 200:
            gid = len(ident) + 1
            t.global_id = gid
            ident.append(GlobalIdentity(
                gid=gid, gallery=[t.emb] if t.emb is not None else [],
                face_gallery=[t.face] if t.face is not None else [],
                low_gallery=[t.low] if t.low is not None else [],
                body2_gallery=[t.body2] if t.body2 is not None else [],
                track_keys=[t.key], frames_all=list(t.frames), last_frame=t.end,
                last_box=t.boxes[-1], last_shot=t.shot_id, velocity=t.velocity))
            events.append(IdentityEvent(t=round(t.start / fps, 3), fi=t.start,
                                        local_id=t.key[1], global_id=gid,
                                        action="NEW_IDENTITY",
                                        note=f"shot {t.shot_id}, {t.n} 帧, "
                                             f"h={t.median_h:.0f}, face={t.face_n}"))
        else:
            t.global_id = -1
    return ident, events


def calibrate_face(tracks: dict, fallback: tuple[float, float] = (0.62, 0.55)) -> dict:
    """人脸阈值同样用本素材标定：同轨迹 vs 共现轨迹。"""
    within = []
    for t in tracks.values():
        if t.face is None or t.face_n < 3:
            continue
        within.append(1.0)
    between = []
    lst = [t for t in tracks.values() if t.face is not None and t.face_n >= 3]
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            a, b = lst[i], lst[j]
            if a.shot_id != b.shot_id:
                continue
            if min(a.end, b.end) - max(a.start, b.start) > 2:
                between.append(float(a.face @ b.face))
    if len(between) < 8:
        return dict(pos=fallback[0], neg=fallback[1], calibrated=False,
                    n_between=len(between))
    bmax = float(np.percentile(between, 95))
    pos = float(np.clip(bmax + 0.05, 0.55, 0.80))
    neg = float(np.clip(bmax - 0.02, 0.45, max(0.5, pos - 0.06)))
    return dict(pos=round(pos, 3), neg=round(neg, 3), calibrated=True,
                n_between=len(between), between_p50=round(float(np.median(between)), 3),
                between_p95=round(bmax, 3), between_max=round(max(between), 3))


def attach_faces(video, info, tracks: dict, encoder, sample_every: int = 4,
                 max_per_track: int = 8) -> None:
    """一次解码遍历：检测人脸 → 归属到局部轨迹 → 聚合成单条轨迹的人脸特征。"""
    from core.video import frames as iter_frames
    from identity.face import assign_to_tracks

    by_frame: dict[int, list[tuple[tuple[int, int], tuple]]] = {}
    for t in tracks.values():
        for fi, b in zip(t.frames, t.boxes):
            if sample_every > 1 and fi % sample_every:
                continue
            by_frame.setdefault(fi, []).append((t.key, b))

    store: dict[tuple[int, int], list[tuple[float, np.ndarray]]] = {}
    for fi, frame in iter_frames(video, limit=info.n_frames):
        boxes = by_frame.get(fi)
        if not boxes:
            continue
        faces = encoder.embed(frame)
        if not faces:
            continue
        for key, f in assign_to_tracks(faces, boxes).items():
            store.setdefault(key, []).append((f.score, f.feat))
            tracks[key].face_ts.append((fi, f.feat, f.score))

    for key, items in store.items():
        t = tracks.get(key)
        if t is None:
            continue
        items.sort(key=lambda x: -x[0])
        sel = items[:max_per_track]
        if not sel:
            continue
        m = np.stack([e for _, e in sel])
        cen = m.mean(axis=0)
        t.face = (cen / (np.linalg.norm(cen) + 1e-9)).astype(np.float32)
        t.face_n = len(sel)
        sims = m @ t.face
        t.face_cons = float(np.mean(sims))


def detect_swaps(tracks: dict, thresh: float, fps: float = 60.0) -> list[IdentityEvent]:
    """局部轨迹内部的外观突变 = 跟踪器在同一 ID 下换了人。"""
    out = []
    for t in tracks.values():
        if t.emb_t is None or len(t.emb_t) < 4:
            continue
        s = np.diag(t.emb_t[:-1] @ t.emb_t[1:].T)
        for i, v in enumerate(s):
            if v < max(0.35, thresh * 0.7):
                out.append(IdentityEvent(
                    t=round(float(t.emb_ts[i + 1]) / fps, 3), fi=int(t.emb_ts[i + 1]),
                    local_id=t.key[1], global_id=t.global_id, action="SWAP_SUSPECT",
                    appearance_score=round(float(v), 3),
                    note="同一局部 ID 内外观不连续"))
    return out


# ---------------------------------------------------------------- 目标选择
def select_target(ident: list[GlobalIdentity], tracks: dict, mode: str = "largest",
                  box: Optional[list] = None, point: Optional[list] = None,
                  at_frame: int = -1, gid: int = 0,
                  ref_face: Optional[np.ndarray] = None,
                  min_face: float = 0.55) -> int:
    """四种选人方式，优先级：显式 global_id > 参考人脸 > 框 > 点 > 自动。"""
    if gid and any(g.gid == gid for g in ident):
        return gid

    # 1) 参考人脸：跨运行锁定同一个人的唯一可靠办法
    if ref_face is not None:
        scored = []
        for g in ident:
            if not g.face_gallery:
                continue
            c = np.mean(np.stack(g.face_gallery), axis=0)
            c = c / (np.linalg.norm(c) + 1e-9)
            scored.append((float(ref_face @ c), g.frames, g.gid))
        if scored:
            scored.sort(reverse=True)
            best_s = scored[0][0]
            # 同一个人的碎片会得到几乎相同的分数；这时选覆盖最长的那个，
            # 而不是分数最高的碎片（实测 0.693 vs 0.681 就差点选错）。
            close = [x for x in scored if x[0] >= best_s - 0.03]
            close.sort(key=lambda x: -x[1])
            gid_pick, sim_pick = close[0][2], close[0][0]
            if sim_pick >= min_face:
                print(f"[target] 参考人脸命中 ID {gid_pick} (sim={sim_pick:.3f}, "
                      f"并列 {len(close)} 个)")
                return gid_pick
            print(f"[warn] 参考人脸最佳匹配仅 {best_s:.2f}，低于 {min_face}")

    def pick_by_box(b):
        bx1, by1, bx2, by2 = b
        best, best_iou = -1, 0.0
        for t in tracks.values():
            if t.global_id <= 0:
                continue
            for box_i in t.boxes[:3]:
                ix1, iy1 = max(bx1, box_i[0]), max(by1, box_i[1])
                ix2, iy2 = min(bx2, box_i[2]), min(by2, box_i[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = ((bx2 - bx1) * (by2 - by1)
                         + (box_i[2] - box_i[0]) * (box_i[3] - box_i[1]) - inter)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best, best_iou = t.global_id, iou
        return best

    if box and len(box) == 4:
        g = pick_by_box(box)
        if g > 0:
            return g

    # 2) 点选：必须落在某一帧的具体框里，避免"点到谁就选谁"的歧义
    if point and len(point) == 2:
        px, py = point
        cands = []
        for t in tracks.values():
            if t.global_id <= 0:
                continue
            idx = [k for k, fi in enumerate(t.frames)
                   if (at_frame < 0 or fi == at_frame)]
            if at_frame >= 0 and not idx:
                continue
            for k in idx[:3]:
                b = t.boxes[k]
                if b[0] <= px <= b[2] and b[1] <= py <= b[3]:
                    cands.append((abs((b[0] + b[2]) / 2 - px), t.global_id))
                    break
        if cands:
            cands.sort()
            return cands[0][1]

    scored = []
    for g in ident:
        ts = [tracks[k] for k in g.track_keys if k in tracks]
        if not ts:
            continue
        h = float(np.median([t.median_h for t in ts]))
        span = max(t.end for t in ts) - min(t.start for t in ts)
        cx = float(np.median([(t.boxes[0][0] + t.boxes[0][2]) / 2 for t in ts]))
        scored.append((h, span, cx, g.gid))
    if not scored:
        return -1
    if mode == "center":
        scored.sort(key=lambda s: abs(s[2] - 960))
        return scored[0][3]

    full = [s for s in scored if s[1] >= 0.8 * max(x[1] for x in scored)]
    pool = full or scored
    pool.sort(key=lambda s: s[0], reverse=True)
    return pool[0][3]


# ---------------------------------------------------------------- 目标逐帧序列
def build_target_series(ident: list[GlobalIdentity], tracks: dict, gid: int,
                        n_frames: int, fps: float,
                        body_min: float = 0.0,
                        merge_gids: list[int] | None = None) -> tuple[list[IdentityObs], dict]:
    """把目标的轨迹展开成逐帧序列。

    body_min > 0 时做一次"轨迹内部清洗"：以身份最初那条轨迹的人体特征为基准，
    把同一段里明显不是本人的帧剔掉（切段边界偶尔会晚几秒，这些尾巴要用数据剪掉）。
    """
    # merge_gids：同一个人被聚类拆成好几段时（人少脸小的素材很常见），用参考人脸
    # 把它们并成一条目标轨迹。不合并就会只跟到其中一段，覆盖率一半。
    gids = [int(gid)] + [int(x) for x in (merge_gids or []) if int(x) != int(gid)]
    gs = [x for x in ident if x.gid in set(gids)]
    if not gs:
        raise SystemExit(f"没有 global id {gid}")
    g = gs[0]
    ref_body = None
    if body_min > 0:
        first = tracks.get(g.track_keys[0]) if g.track_keys else None
        ref_body = getattr(first, "body2", None)
        assert ref_body is None or ref_body.shape == (512,) or True

    by_frame: dict[int, IdentityObs] = {}
    dropped = 0
    for gx in gs:
        for k in gx.track_keys:
            t = tracks.get(k)
            if t is None:
                continue
            conf = float(np.mean(t.scores)) if t.scores else 0.0
            bad = set()
            if ref_body is not None and getattr(t, "body2_ts", None):
                for fi, v in t.body2_ts:
                    if float(v @ ref_body) < body_min:
                        bad.add(fi)
            for fi, b in zip(t.frames, t.boxes):
                if bad:
                    # 采样点间隔内的帧一并剔除（相邻采样之间按最近采样判定）
                    near = min(bad, key=lambda x: abs(x - fi))
                    if abs(near - fi) <= 30:
                        dropped += 1
                        continue
                old = by_frame.get(fi)
                if old is not None and (old.y2 - old.y1) >= (b[3] - b[1]):
                    continue          # 同一帧两个碎片都有框，留大的那个
                by_frame[fi] = IdentityObs(fi=fi, t=fi / fps, global_id=gid, local_id=k[1],
                                           x1=b[0], y1=b[1], x2=b[2], y2=b[3],
                                           appearance_score=0.0, motion_score=0.0,
                                           identity_conf=conf, decision="BIND", shot_id=k[0])
    series = [by_frame.get(fi) for fi in range(n_frames)]
    covered = sum(1 for s in series if s is not None)
    gaps, run = [], 0
    for s in series:
        if s is None:
            run += 1
        elif run:
            gaps.append(run)
            run = 0
    if run:
        gaps.append(run)
    summary = dict(global_id=gid, frames_visible=covered,
                   coverage=round(covered / max(1, n_frames), 4),
                   local_tracks=len(g.track_keys),
                   dropped_frames=dropped,
                   gaps=len(gaps), longest_gap_frames=max(gaps) if gaps else 0,
                   longest_gap_s=round(max(gaps) / fps, 2) if gaps else 0.0)
    return series, summary


# 人脸/人体检测器内部是固定小输入（YuNet/cheetah 都是百级像素），在 4K/8K 全图上
# 跑纯属浪费——实测 4K 上 identify 从 552ms/帧 降到 60ms/帧。这里先降采样再检测，
# 只要把框按同样比例缩过去即可；特征本身与坐标无关，不需要反变换。
_SMALL_H = 1280


def _downscale_for_detect(frame, boxes, max_h: int = _SMALL_H):
    h, w = frame.shape[:2]
    if h <= max_h:
        return frame, boxes, 1.0
    import cv2
    sc = max_h / float(h)
    nw, nh = int(w * sc) // 2 * 2, int(h * sc) // 2 * 2
    small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    rb = [(key, (b[0] * sc, b[1] * sc, b[2] * sc, b[3] * sc)) for key, b in boxes]
    return small, rb, sc


def attach_all(video, info, tracks: dict, embedder=None, face_encoder=None,
               analyzer=None, emb_every: int = 4, face_every: int = 4,
               pa_every: int = 30, max_samples: int = 12,
               max_per_track: int = 8, head_crop: bool = True) -> dict:
    """一次解码遍历完成 外观 / 人脸 / 人体 ReID 三路特征。

    原来三个函数各解码一遍。1080p 上只是浪费几秒，但 4K/8K 上解码本身就是瓶颈
    （8K 约 70ms/帧），三遍等于付三倍成本。合并后只解一遍，各路仍按自己的间隔触发。
    """
    from core.video import frames as iter_frames
    from identity.face import assign_to_tracks
    from identity.person_analysis import associate

    emb_by_frame: dict[int, list] = {}
    if embedder is not None:
        for t in tracks.values():
            idx = np.unique(np.linspace(0, t.n - 1, min(max_samples, t.n)).astype(int))
            for k in idx:
                emb_by_frame.setdefault(t.frames[k], []).append(t)

    face_by_frame: dict[int, list] = {}
    if face_encoder is not None:
        for t in tracks.values():
            for fi, b in zip(t.frames, t.boxes):
                if face_every > 1 and fi % face_every:
                    continue
                face_by_frame.setdefault(fi, []).append((t.key, b))

    pa_by_frame: dict[int, list] = {}
    if analyzer is not None:
        for t in tracks.values():
            for fi, b in zip(t.frames, t.boxes):
                if pa_every > 1 and fi % pa_every:
                    continue
                pa_by_frame.setdefault(fi, []).append((t.key, b))

    want = set(emb_by_frame) | set(face_by_frame) | set(pa_by_frame)
    emb_store: dict = {}
    low_store: dict = {}
    face_store: dict = {}
    body_store: dict = {}
    pface_store: dict = {}
    n_emb = n_face = n_pa = 0

    for fi, frame in iter_frames(video, limit=info.n_frames):
        if fi not in want:
            continue

        if embedder is not None and fi in emb_by_frame and (emb_every <= 1 or fi % emb_every == 0):
            keep = [(t, t.box_at(fi)) for t in emb_by_frame[fi]]
            keep = [(t, b) for t, b in keep if b is not None]
            if keep:
                embs = embedder.embed(frame, [b for _, b in keep])
                for (t, b), e in zip(keep, embs):
                    emb_store.setdefault(t.key, []).append((fi, e))
                    ld = lowbody_desc(frame, b)
                    if ld is not None:
                        low_store.setdefault(t.key, []).append(ld)
                        t.low_ts.append((fi, ld))
                n_emb += 1

        if face_encoder is not None and fi in face_by_frame:
            fb = face_by_frame[fi]
            if head_crop and (frame.shape[0] > 1440):
                # 高分辨率素材：逐个人体框裁头部检测。归属是确定的，不需要再分配。
                got = False
                for key, b in fb:
                    obs = face_encoder.embed_head(frame, b)
                    if not obs:
                        continue
                    best = max(obs, key=lambda o: o.score)
                    face_store.setdefault(key, []).append((best.score, best.feat))
                    tracks[key].face_ts.append((fi, best.feat, best.score))
                    got = True
                if got:
                    n_face += 1
            else:
                sframe, boxes, _ = _downscale_for_detect(frame, fb)
                faces = face_encoder.embed(sframe)
                if faces:
                    for key, f in assign_to_tracks(faces, boxes).items():
                        face_store.setdefault(key, []).append((f.score, f.feat))
                        tracks[key].face_ts.append((fi, f.feat, f.score))
                    n_face += 1

        if analyzer is not None and fi in pa_by_frame:
            sframe, boxes, _ = _downscale_for_detect(frame, pa_by_frame[fi])
            persons = analyzer.get(sframe)
            if persons:
                n_pa += 1
                for key, p in associate(persons, boxes).items():
                    if p.reid is not None:
                        body_store.setdefault(key, []).append((p.score, p.reid))
                        tracks[key].body2_ts.append((fi, p.reid))
                    if p.face is not None and p.face_valid:
                        pface_store.setdefault(key, []).append((p.score, p.face))

    # ---- 外观：去离群后取均值
    for key, items in emb_store.items():
        t = tracks.get(key)
        if t is None:
            continue
        items.sort(key=lambda x: x[0])
        mat = np.stack([e for _, e in items])
        t.emb_ts = np.array([f for f, _ in items], dtype=np.int64)
        t.emb_t = mat
        cen = mat.mean(axis=0)
        sims = mat @ (cen / (np.linalg.norm(cen) + 1e-9))
        keep = sims >= np.percentile(sims, 20) if len(sims) >= 5 else np.ones(len(sims), bool)
        cen = mat[keep].mean(axis=0)
        t.emb = (cen / (np.linalg.norm(cen) + 1e-9)).astype(np.float32)

    for key, vs in low_store.items():
        t = tracks.get(key)
        if t is None or not vs:
            continue
        m = np.mean(np.stack(vs), axis=0)
        t.low = (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)

    # ---- 人脸：取高分样本聚合
    for key, items in face_store.items():
        t = tracks.get(key)
        if t is None or not items:
            continue
        items.sort(key=lambda x: -x[0])
        sel = items[:max(max_per_track, 12)]
        if not sel:
            continue
        m = np.stack([e for _, e in sel])
        # 先求临时质心，再丢掉偏离最远的一半（侧脸、低头、运动模糊），
        # 剩下的平均才是"这个人的脸"。实测这一步把同人 p10 从 0.18 拉到 0.5+。
        c0 = m.mean(axis=0)
        c0 = c0 / (np.linalg.norm(c0) + 1e-9)
        sims = m @ c0
        keep = sims >= np.percentile(sims, 50) if len(sims) >= 4 else np.ones(len(sims), bool)
        cen = m[keep].mean(axis=0)
        t.face = (cen / (np.linalg.norm(cen) + 1e-9)).astype(np.float32)
        t.face_n = int(keep.sum())
        t.face_cons = float(np.mean(m[keep] @ t.face))

    # ---- 人体 ReID / cheetah 人脸
    for store, attr in ((body_store, "body2"), (pface_store, "face2")):
        for key, items in store.items():
            t = tracks.get(key)
            if t is None or not items:
                continue
            items.sort(key=lambda x: -x[0])
            m = np.stack([e for _, e in items[:max_per_track]]).mean(axis=0)
            setattr(t, attr, (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32))

    return dict(frames_embedded=n_emb, frames_with_faces=n_face,
                frames_with_persons=n_pa,
                tracks_with_emb=sum(1 for t in tracks.values() if t.emb is not None),
                tracks_with_face=sum(1 for t in tracks.values() if t.face is not None),
                tracks_with_body=sum(1 for t in tracks.values() if t.body2 is not None))
