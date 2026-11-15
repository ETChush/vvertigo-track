"""参考人脸驱动的单目标定位。

为什么不直接用跟踪轨迹
--------------------
多人、频繁转身、机位还在变焦的素材上，跟踪器的长轨迹一定会串人。1.mp4 的实测：
同一条轨迹在 5.9s 是左边某个人（人脸相似度 0.044），120s 是她（0.709），
183s 又是别人。事后拿人脸去核验只能"发现"错，不能把它变对。

所以这里换一条路，只相信两件事：
  1. 脸转过来了的那些帧 —— 人脸是最硬的证据
  2. 相邻 1.5 秒内的框位置连续性 —— 转身背对时，框还在原地

做法：稀疏撒人脸锚点 → 锚点之间用检测框做短距离关联补满 → 更长的空洞留空，
交给相机 HOLD。宁可短暂保持原取景，也不要跟着别人走。

好处是**不需要长轨迹**，只要每帧的检测框。所以可以配合 BoT-SORT（比 TAQ 快
6-8 倍），框还更准（1280 检测 vs TAQ 的 720 代理）。
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from core.schema import IdentityObs


def _box_cx(b) -> float:
    return (b[0] + b[2]) / 2.0


def _box_h(b) -> float:
    return max(1.0, b[3] - b[1])


def ref_hash(ref_feat: np.ndarray) -> str:
    return hashlib.md5(np.asarray(ref_feat, dtype=np.float32).tobytes()).hexdigest()[:10]


def scan(video: str, tracks: dict, ref_feat: np.ndarray, *, every: int = 8,
         top_k: int = 10, up: int = 1, log=print,
         resume: list[dict] | None = None,
         on_progress=None) -> list[dict]:
    """逐帧在所有检测框上比对参考人脸，返回稀疏锚点。

    两个关键开销决定：
      - 帧在 ffmpeg 侧抽（select），4K 整片走管道要 37.7 FPS，只取每 8 帧后
        同样的事 21 秒就跑完。
      - 头部**不放大**（up=1）。实测放大 2 倍慢 3 倍（13.6 → 4.4 帧/秒），
        命中率却没有提升（8% vs 7%）——4K 素材的脸本来就有 170px。

    resume：断点续跑时传入上一轮已扫出的锚点；函数跳过它们已经扫过的帧号，
    只对新采样帧做比对，最后把新旧结果合并返回。
    """
    from core.fastvideo import pipe_frames, pipe_frames_step, probe_size

    from identity.face import FaceEncoder

    byf: dict[int, list] = defaultdict(list)
    for t in tracks.values():
        for fi, b in zip(t.frames, t.boxes):
            byf[fi].append(b)
    if not byf:
        return []

    # 断点续跑：resume 里已扫过的帧号直接跳过（管道仍需顺序解到那，但跳过了
    # 最耗时的 embed_head 比对，续跑仍能省下大头）
    done_fi = set()
    out: list[dict] = []
    if resume:
        out = list(resume)
        done_fi = {int(r["fi"]) for r in resume}

    # 抽帧方式要按帧率选。ffmpeg 的 select 在 120fps 的素材上会丢掉约 3/4 的帧
    # （实测 exid.mp4：期望 2932 帧，只输出 733 帧），帧号对不上就全盘 miss，
    # 最后 0 个锚点。高帧率直接顺序解码、在 Python 侧过滤，慢一点但一帧不少。
    _, _, vfps = probe_size(video)
    # 采样间隔按 60fps 折算：这里关心的是"每 0.13 秒看一次脸"，不是"每 8 帧"。
    # 120fps 的 every=8 实际采样密度是 60fps 的两倍，时间直接翻倍还多余。
    if vfps > 65:
        eff = max(1, int(round(every * vfps / 60.0)))
        if eff != every:
            print(f"  [anchor] {vfps:.0f}fps，采样间隔按 60fps 折算：{every} → {eff}")
            every = eff
        print(f"  [anchor] {vfps:.0f}fps 素材，顺序解码（select 在这种帧率下会丢帧）")
        it = ((fi, f) for fi, f in pipe_frames(video) if fi % every == 0)
    else:
        it = ((k * every, f) for k, f in pipe_frames_step(video, every))

    # 估算总采样帧数，用于进度和 ETA。以检测框覆盖到的最大帧号为上界。
    _max_fi = max(byf) if byf else 0
    total_samples = max(1, _max_fi // every + 1)

    enc = FaceEncoder()
    n = 0
    t_scan = time.time()
    for fi, frame in it:
        if fi in done_fi:
            continue                    # 断点续跑：已扫过
        bs = byf.get(fi)
        if not bs:
            continue
        bs = sorted(bs, key=lambda b: -_box_h(b))[:top_k]
        best = (-1.0, None, None)
        for b in bs:
            obs = enc.embed_head(frame, b, up=up)
            if not obs:
                continue
            o = max(obs, key=lambda x: x.score)
            sc = float(o.feat @ ref_feat)
            if sc > best[0]:
                best = (sc, b, float(o.h))
        if best[1] is not None:
            out.append(dict(fi=fi, sim=round(best[0], 4),
                            x1=round(float(best[1][0]), 1), y1=round(float(best[1][1]), 1),
                            x2=round(float(best[1][2]), 1), y2=round(float(best[1][3]), 1),
                            face_h=round(best[2], 1)))
        n += 1
        if n % 300 == 0:
            el = time.time() - t_scan
            rate = n / max(el, 1e-6)
            remain = max(0, total_samples - n)
            eta = f"  剩约 {remain / rate / 60:.1f} 分钟" if remain > 0 else ""
            log(f"  [anchor] 已扫描 {n}/{total_samples} 采样帧  {rate:.1f} 帧/秒{eta}")
            if on_progress is not None:
                on_progress(out)
    return out


def scan_cached(video: str, tracks: dict, ref_feat: np.ndarray, cache: Path, *,
                every: int = 15, up: int = 1, top_k: int = 10,
                log=print) -> list[dict]:
    """带缓存的人脸扫描。参考图没变就复用；中断过就从断点续跑。"""
    if cache.exists():
        try:
            rows = json.loads(cache.read_text(encoding="utf-8"))
            if rows and rows[0].get("ref") == ref_hash(ref_feat) \
                    and rows[0].get("every") == every \
                    and rows[0].get("up", 1) == up \
                    and rows[0].get("top_k", 10) == top_k:
                log(f"  [anchor] 复用扫描结果 {len(rows) - 1} 帧（{cache.name}）")
                return [r for r in rows[1:]]
        except (ValueError, KeyError, IndexError):
            pass

    # 断点续跑：上一轮扫描中途被中断，.partial 里存了已扫出的锚点，接着扫。
    partial = cache.with_suffix(cache.suffix + ".partial")
    resume = None
    if partial.exists():
        try:
            resume = json.loads(partial.read_text(encoding="utf-8"))
            if resume and resume[0].get("ref") == ref_hash(ref_feat) \
                    and resume[0].get("top_k", 10) == top_k:
                resume = resume[1:]
                log(f"  [anchor] 断点续跑：已有 {len(resume)} 个锚点，接着扫")
            else:
                resume = None
        except (ValueError, KeyError, IndexError):
            resume = None

    def _dump(out_rows):
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(
            [dict(ref=ref_hash(ref_feat), every=every, up=up, top_k=top_k)]
            + out_rows, ensure_ascii=False), encoding="utf-8")

    rows = scan(video, tracks, ref_feat, every=every, up=up, top_k=top_k,
                log=log, resume=resume, on_progress=_dump)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([dict(ref=ref_hash(ref_feat), every=every, up=up,
                                      top_k=top_k)]
                                + rows, ensure_ascii=False), encoding="utf-8")
    if partial.exists():
        partial.unlink()                # 完成，清掉断点文件
    return rows


def build_series(tracks: dict, n_frames: int, fps: float, anchors: list[dict], *,
                 thr: float = 0.55, max_gap: int = 150, max_grow: int = 60,
                 cost_max: float = 0.11, gid: int = 0) -> tuple[list, dict]:
    """锚点 + 短距离框关联，生成逐帧目标框序列。

    max_gap   两个锚点相隔超过这么多帧就不桥接（留空 → 相机 HOLD）
    max_grow  单次生长最多补这么多帧，防止长时间靠惯性漂走
    cost_max  关联代价门限 = 预测位移 / 框高；0.14 约等于 0.14 个身高
    """
    pool: dict[int, list] = defaultdict(list)
    for t in tracks.values():
        for fi, b in zip(t.frames, t.boxes):
            pool[fi].append(b)

    series: list = [None] * n_frames
    keep = [a for a in anchors if a["sim"] >= thr]
    for a in keep:
        series[int(a["fi"])] = (a["x1"], a["y1"], a["x2"], a["y2"])
    keys = sorted(int(a["fi"]) for a in keep)
    if not keys:
        return series, dict(anchors=len(anchors), hits=0, filled=0, ratio=0.0,
                            mean_sim=0.0, hit_ratio=0.0)

    def grow(start: int, step: int, floor: int, ceil: int) -> int:
        b = series[start]
        cx, h = _box_cx(b), _box_h(b)
        vx = 0.0
        fi = start + step
        n = 0
        while 0 <= fi < n_frames and n < max_grow and floor <= fi <= ceil:
            if series[fi] is not None:          # 撞到另一个锚点，停
                break
            cands = pool.get(fi)
            if not cands:
                break
            pred = cx + vx
            ar0 = (b[2] - b[0]) / h
            best, bc = None, 1e9
            for c in cands:
                cw = _box_cx(c)
                ch = _box_h(c)
                car = (c[2] - c[0]) / ch
                # 位置连续性 + 尺寸连续性 + 形状连续性。最后一项是补的教训：
                # 1.mp4 有一帧的框宽到装下两个人（金发 + 黑发），位置和高度都对，
                # 只有宽高比不对，结果整段跟到了旁边那位。人的宽高比在相邻帧间
                # 几乎不变，突变基本就是框合并或换人了。
                cost = (abs(cw - pred) / h
                        + 0.6 * abs(float(np.log(ch / h)))
                        + 0.5 * abs(float(np.log(max(car, 1e-3) / max(ar0, 1e-3)))))
                if cost < bc:
                    bc, best = cost, c
            if best is None or bc > cost_max:
                break
            ncx, nh = _box_cx(best), _box_h(best)
            vx = 0.7 * vx + 0.3 * (ncx - cx)
            series[fi] = (float(best[0]), float(best[1]), float(best[2]), float(best[3]))
            cx, h = ncx, nh
            fi += step
            n += 1
        return n

    for a, b in zip(keys, keys[1:]):
        if b - a <= 1 or b - a > max_gap:
            continue
        grow(a, +1, a, b)
        grow(b, -1, a, b)
    grow(keys[0], -1, 0, keys[0])
    grow(keys[-1], +1, keys[-1], n_frames - 1)

    conf = {int(a["fi"]): float(a["sim"]) for a in keep}
    filled = 0
    for fi in range(n_frames):
        if series[fi] is None:
            continue
        b = series[fi]
        series[fi] = IdentityObs(fi=fi, t=fi / fps, global_id=gid, local_id=-1,
                                 x1=b[0], y1=b[1], x2=b[2], y2=b[3],
                                 appearance_score=0.0, motion_score=0.0,
                                 identity_conf=conf.get(fi, 0.0),
                                 decision=("FACE" if fi in conf else "GROW"),
                                 shot_id=0)
        filled += 1

    sims = [a["sim"] for a in keep]
    return series, dict(anchors=len(anchors), hits=len(keep), filled=filled,
                        ratio=filled / max(1, n_frames),
                        mean_sim=round(float(np.mean(sims)), 3) if sims else 0.0,
                        hit_ratio=len(keep) / max(1, len(anchors)))


def summarize(series: list, fps: float) -> dict:
    """可见段的粗略统计，用于判断"跟丢了还是人不在"。"""
    n = len(series)
    gaps, cur = [], 0
    for s in series:
        if s is None:
            cur += 1
        else:
            if cur:
                gaps.append(cur)
            cur = 0
    if cur:
        gaps.append(cur)
    gaps.sort(reverse=True)
    return dict(visible=sum(1 for s in series if s is not None) / max(1, n),
                longest_gap_s=round(gaps[0] / fps, 2) if gaps else 0.0,
                n_gaps=len([g for g in gaps if g > fps]))
