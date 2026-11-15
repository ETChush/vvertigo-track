# -*- coding: utf-8 -*-
"""VVERTIGO 一键直拍：给一个目录，每个人一套（人脸图 / 竖屏直拍 / 点位 CSV）。

用法：
  python run.py --src 待处理
  python run.py --src 待处理 --max-targets 5 --preset steady
  python run.py --src 待处理 --zoom 1.2 --prefix A
  python run.py --src 待处理 --targets 1,3          # 只出排名第 1、3 的人
  python run.py --src 待处理 --redo-all              # 连检测一起重跑

产物：
  outputs/<视频名>/
    ID1/A-ID1.jpg   该 ID 的人脸图（按质量挑，不是第一帧）
    ID1/A-ID1.mp4   竖屏直拍，zoom=1.0 不额外放大，保留原音轨
    ID1/A-ID1.csv   点位（供 ae/VVERTIGO_Track.jsx 导入 AE 关键帧）
    ID2/ ...
    manifest.json   每个 ID 的覆盖/全身率/关键帧/误差，判断哪条要重跑
    _work/          中间产物，断点续跑靠它，可以删

为什么不做"统一压 1080 分析"：实测 4K 上解码只占 7%（259ms/帧里 17.5ms），
检测和特征才是瓶颈，代理流省不到东西却多一层坐标换算的风险。这一步不做。
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from camera import planner                              # noqa: E402
from core import config as cfgmod                        # noqa: E402
from core.schema import IdentityObs, read_json, read_jsonl, write_json  # noqa: E402
from core.video import probe, read_frame_at              # noqa: E402
from export import points as pts                         # noqa: E402
from export import render                                # noqa: E402
from identity import face_anchor, manager                             # noqa: E402
from pipeline import analyze, detect_fast, identify                   # noqa: E402

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm")


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fa5]+", "_", name)
    return s.strip("_")[:32]


def load_identities(out_dir: Path):
    """从 tracks_kept.jsonl 重建 global id → 局部轨迹（切段后的最终版本）。"""
    kept = list(read_jsonl(out_dir / "tracks_kept.jsonl"))
    tracks = manager.build_local_tracks(kept)
    for r in kept:
        k = (int(r["sid"]), int(r["lid"]))
        if k in tracks:
            tracks[k].global_id = int(r["gid"])
    by_gid: dict[int, list] = {}
    for k, t in tracks.items():
        if t.global_id > 0:
            by_gid.setdefault(t.global_id, []).append(t)
    faces = {}
    fp = out_dir / "identity_faces.json"
    if fp.exists():
        faces = read_json(fp)
    ident = []
    for gid, ts in by_gid.items():
        f = faces.get(str(gid))
        ident.append(manager.GlobalIdentity(
            gid=gid, gallery=[], track_keys=[t.key for t in ts],
            frames_all=[fi for t in ts for fi in t.frames],
            face_gallery=[np.array(f, dtype=np.float32)] if f else []))
    ident.sort(key=lambda g: -len(g.frames_all))
    return tracks, ident, by_gid


# ---------------------------------------------------------------- 选人
def pick_candidates(ident, tracks, info, n_frames: int, args) -> tuple[list, list]:
    """谁值得单独出一条。

    用户的口径"首帧像素大的要、小的不要"落地成两级：
      1. 硬门槛（绝对尺寸 + 覆盖率）：框高 < 画面高 20% 的一律判为观众/背景；
      2. 相对门槛（框高 / 画面最高者）：远景观众和台上演员的绝对像素能差 2 倍，
         只用绝对阈值在 4K 上分不开。
    过了门槛的按 score 排序，score 用"窗口打分"而不是首帧：
      面积p90 × 覆盖时长 × 居中度 × 人脸可用率
    p90 而不是均值，是为了抗"某一帧突然特写"和检测框抖动。
    """
    W, H = float(info["width"]), float(info["height"])
    fps = float(info["fps"])
    rows = []
    for g in ident:
        ts = [tracks[k] for k in g.track_keys if k in tracks]
        if not ts:
            continue
        hs = np.array([b[3] - b[1] for t in ts for b in t.boxes], dtype=float)
        if not hs.size:
            continue
        cxs = np.array([(b[0] + b[2]) / 2.0 for t in ts for b in t.boxes], dtype=float)
        frames: set[int] = set()
        for t in ts:
            frames.update(t.frames)
        cov = len(frames) / max(1, n_frames)
        h_p90 = float(np.percentile(hs, 90))
        cx_med = float(np.median(cxs))
        center = 1.0 - min(1.0, abs(cx_med - W / 2.0) / max(1.0, W / 2.0))
        face_ok = 1.0 if g.face_gallery else 0.35
        rows.append(dict(
            gid=g.gid, h_p90=h_p90, h_med=float(np.median(hs)), cov=cov,
            center=center, face_ok=face_ok, n_tracks=len(ts),
            t0=round(min(frames) / fps, 1), t1=round(max(frames) / fps, 1),
            score=(h_p90 / H) * cov * (0.55 + 0.45 * center) * face_ok, rank=0))

    gate_abs = H * float(args.min_height)
    keep = [r for r in rows if r["h_p90"] >= gate_abs and r["cov"] >= args.min_coverage]
    kept_gid = {r["gid"] for r in keep}
    dropped = [r for r in rows if r["gid"] not in kept_gid]
    for r in dropped:
        r["reason"] = ("框高不足" if r["h_p90"] < gate_abs else "覆盖不足")

    if keep:
        hmax = max(r["h_p90"] for r in keep)
        floor = float(args.height_ratio) * hmax
        far = [r for r in keep if r["h_p90"] < floor]
        for r in far:
            r["reason"] = "远景/次要（框高不足最高的 %.0f%%）" % (args.height_ratio * 100)
        dropped.extend(far)
        keep = [r for r in keep if r["h_p90"] >= floor]

    keep.sort(key=lambda r: -r["score"])
    for i, r in enumerate(keep):
        r["rank"] = i + 1
    return keep, dropped


def _ref_feature(path: str) -> np.ndarray:
    """从一张参考图里取人脸特征。

    先按"整张图里找人脸"走（用户随手截的图往往带背景），检不到再退化成
    "整张图就是一张对齐好的人脸"。两条路都不需要用户自己裁。
    """
    import cv2

    q = _resolve(path, must_exist=False)
    # cv2.imread 在 Windows 上不认非 ASCII 路径，含中文的目录会直接返回 None。
    # 走 fromfile + imdecode 就没有这个问题。
    img = None
    try:
        buf = np.fromfile(str(q), dtype=np.uint8)
        if buf.size:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except OSError:
        img = None
    if img is None:
        raise SystemExit(f"--ref-face 读不到图片：{q}")
    enc = _face_encoder()
    obs = enc.embed(img)
    if obs:
        # 图里往往不止一张脸（旁边还有人）。按**脸的大小**取，不是按检测分——
        # 检测分几乎都是 0.92，同分时顺序不定，实测就取到了旁边戴眼镜的男生，
        # 于是整片相似度全 0。主角通常离镜头最近，脸最大。
        obs.sort(key=lambda o: -o.h)
        pick = obs[0]
        if len(obs) > 1:
            others = ", ".join(f"{o.h:.0f}px" for o in obs[1:4])
            print(f"  参考图里检出 {len(obs)} 张脸，取最大的 {pick.h:.0f}px"
                  f"（其余 {others}）")
        return np.asarray(pick.feat, dtype=np.float32)
    return np.asarray(enc.embed_aligned(img), dtype=np.float32)


_REF_FEAT = None


def _series_by_face(tracks: dict, video: str, n_frames: int, fps: float, ref_feat,
                    every: int = 8, thr: float = 0.55, top_k: int = 6,
                    fill_gap: int = 20, gid: int = 0) -> tuple[list, dict]:
    """不靠聚类，直接用人脸在每一帧的检测框里认人。

    多人、频繁转身、妆容相近的素材上，聚类一定会把别人并进来。但只要目标转过脸来，
    人脸就是最硬的证据。这里逐帧问一句"这些人里哪个长得像参考图"，
    取最像且过阈值的那一个，绕开整条聚类链路。
    """
    from collections import defaultdict

    from core.video import frames as iter_frames

    by_frame: dict[int, list] = defaultdict(list)
    for t in tracks.values():
        for fi, b in zip(t.frames, t.boxes):
            if every > 1 and fi % every:
                continue
            by_frame[fi].append(b)

    enc = _face_encoder()
    hits: dict[int, tuple] = {}
    for fi, frame in iter_frames(video, limit=n_frames):
        bs = by_frame.get(fi)
        if not bs:
            continue
        bs = sorted(bs, key=lambda b: -(b[3] - b[1]))[:top_k]
        best_s, best_b = -1.0, None
        for b in bs:
            obs = enc.embed_head(frame, (b[0], b[1], b[2], b[3]))
            if not obs:
                continue
            o = max(obs, key=lambda x: x.score)
            s = float(o.feat @ ref_feat)
            if s > best_s:
                best_s, best_b = s, b
        if best_b is not None and best_s >= thr:
            hits[fi] = (best_b, best_s)

    series = [None] * n_frames
    for fi, (b, sc) in hits.items():
        series[fi] = IdentityObs(fi=fi, t=fi / fps, global_id=gid, local_id=-1,
                                 x1=b[0], y1=b[1], x2=b[2], y2=b[3],
                                 appearance_score=sc, motion_score=0.0,
                                 identity_conf=sc, decision="FACE", shot_id=0)

    # 短空洞用两端线性插值补齐，避免每 8 帧就断一次让相机一直 HOLD
    i = 0
    while i < n_frames:
        if series[i] is not None:
            i += 1
            continue
        j = i
        while j < n_frames and series[j] is None:
            j += 1
        if i > 0 and j < n_frames and (j - i) <= fill_gap:
            a, b2 = series[i - 1], series[j]
            for k in range(i, j):
                u = (k - i + 1) / (j - i + 1)
                series[k] = IdentityObs(
                    fi=k, t=k / fps, global_id=gid, local_id=-1,
                    x1=a.x1 + (b2.x1 - a.x1) * u, y1=a.y1 + (b2.y1 - a.y1) * u,
                    x2=a.x2 + (b2.x2 - a.x2) * u, y2=a.y2 + (b2.y2 - a.y2) * u,
                    appearance_score=0.0, motion_score=0.0,
                    identity_conf=0.0, decision="INTERP", shot_id=0)
        i = j

    got = sum(1 for s2 in series if s2 is not None)
    return series, dict(hits=len(hits), filled=got, ratio=got / max(1, n_frames),
                        mean_sim=round(float(np.mean([v[1] for v in hits.values()])), 3)
                        if hits else 0.0)


def _verify_series_face(series: list, video: str, ref_feat, thr: float = 0.50,
                        every: int = 15, pad: int = 15) -> dict:
    """拿参考人脸逐段核验目标轨迹，把"画面里其实是别人"的段落剔掉。

    聚类在这一类素材上一定会混进别人（妆容相近、同款服装），光靠覆盖率排序看不出来。
    这里直接在像素上对照：每隔 15 帧抽一帧，按当前框裁头部提人脸，和参考图比。
    检不到脸（侧脸、低头）不算脏，只处理"明确不是这个人"的帧。
    """
    from core.video import frames as iter_frames

    n = len(series)
    want = {i for i in range(0, n, every) if series[i] is not None}
    if not want or ref_feat is None:
        return dict(checked=0, rejected=0)

    enc = _face_encoder()
    bad = []
    for fi, frame in iter_frames(video, limit=n):
        if fi not in want:
            continue
        o = series[fi]
        obs = enc.embed_head(frame, (o.x1, o.y1, o.x2, o.y2))
        if not obs:
            continue
        best = max(obs, key=lambda x: x.score)
        if float(best.feat @ ref_feat) < thr:
            bad.append(fi)

    for fi in bad:
        for k in range(max(0, fi - pad), min(n, fi + pad + 1)):
            series[k] = None
    return dict(checked=len(want), rejected=len(bad))


def _lock_target(cands: list, faces_path: Path, args) -> list:
    """把"跟谁"从排名改成可指定，并把锁定的那位重排成 ID1。

    ID 编号本来就只是打分排名。seg_002 上第一名 97.0%、第二名 96.9%，
    差 0.1% 就可能翻——用户按 ID1 去比上一次的 ID2，自然会觉得"算法变差了"。
    所以给出两条确定性的路：--gid（同一条视频内）和 --ref-face（跨视频用同一张脸）。
    """
    global _REF_FEAT
    target = None
    if args.gid:
        hit = [r for r in cands if int(r["gid"]) == int(args.gid)]
        if not hit:
            raise SystemExit(f"--gid {args.gid} 不在候选里；可选："
                             + ", ".join(f"gid{r['gid']}(ID{r['rank']})" for r in cands))
        target = hit[0]
    elif args.ref_face:
        feat = _ref_feature(str(args.ref_face))
        _REF_FEAT = feat
        faces = read_json(faces_path) if faces_path.exists() else {}
        best, best_sim = None, -1.0
        for r in cands:
            c = faces.get(str(r["gid"]))
            if not c:
                continue
            sim = float(feat @ np.array(c, dtype=np.float32))
            if sim > best_sim:
                best, best_sim = r, sim
        if best is None:
            print("[warn] --ref-face 用不上：本次没有任何 ID 拿到可用人脸，退回排名")
        elif best_sim < 0.35:
            print(f"[warn] --ref-face 最相似只有 {best_sim:.2f}，不像同一个人，"
                  f"退回排名（换一张更正的脸图试试）")
        else:
            print(f"  参考人脸 → gid{best['gid']}（原 ID{best['rank']}，相似度 {best_sim:.2f}）")
            target = best
            # 同一个人常被聚类拆成好几段（人脸 0.72 / 0.72 / 0.71，别人只有 0.41）。
            # 只锁一段就是半个视频跟丢，所以把"人脸差在 0.05 以内"的碎片一起纳入。
            merge = []
            for r in cands:
                if r is best:
                    continue
                c = faces.get(str(r["gid"]))
                if not c:
                    continue
                s2 = float(feat @ np.array(c, dtype=np.float32))
                if s2 >= best_sim - 0.05:
                    merge.append(int(r["gid"]))
            best["merge_gids"] = merge
            if merge:
                print(f"  同一人的其他碎片一并纳入：gid{merge}（人脸相似度与最佳相差 ≤0.05）")

    if target is None:
        return cands

    order = [target] + [r for r in cands if r["gid"] != target["gid"]]
    for i, r in enumerate(order):
        r["rank"] = i + 1
    print(f"  已锁定 gid{target['gid']} 为 ID1，其余按原分数顺延")
    return order


# ---------------------------------------------------------------- 人脸图
_FACE_ENC = None


def _face_encoder():
    global _FACE_ENC
    if _FACE_ENC is None:
        from identity.face import FaceEncoder
        _FACE_ENC = FaceEncoder()
    return _FACE_ENC


def export_face(video: str, series, out_jpg: Path, face_ref, n_try: int = 12):
    """挑一张该 ID 的人脸存成 jpg。

    不取"第一张"：第一张常常是侧脸或运动模糊，做参考图既难看也不好用。
    也不复用 identify 阶段的采样：那批特征是轨迹级聚合，没存人脸框，而且
    load_identities 重建出来的轨迹里没有那些字段。这里按"目标可见 + 框大"的帧
    重新检测一遍，再用人脸质心核验，防止裁到隔壁舞者的脸（交叉段落真实存在）。
    """
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    vis = [(i, s) for i, s in enumerate(series) if s is not None]
    if not vis:
        return dict(found=False, frame=None, face_h=None, sim=None,
                    note="目标从不可见"), None

    # 先取框最大的 60 帧（人越大脸越大），再按时间均匀抽 n_try 帧，避免全挤在一处
    vis.sort(key=lambda x: -(x[1].y2 - x[1].y1))
    pool = sorted(vis[:60], key=lambda x: x[0])
    step = max(1, len(pool) // n_try)
    picks = pool[::step][:n_try]

    enc = _face_encoder()
    best = None                      # (score, sim, frame, obs, fi)
    for fi, s in picks:
        frame = read_frame_at(video, fi)
        if frame is None:
            continue
        obs = enc.embed_head(frame, (s.x1, s.y1, s.x2, s.y2))
        if not obs:
            continue
        for o in obs:
            sim = float(o.feat @ face_ref) if face_ref is not None else 1.0
            sc = o.score * (0.5 + max(0.0, sim)) * min(1.0, o.h / 80.0)
            if best is None or sc > best[0]:
                best = (sc, sim, frame, o, fi)
        if best is not None and best[1] >= 0.55 and best[3].h >= 70:
            break                    # 已经够好，不必再解码

    if best is not None and (face_ref is None or best[1] >= 0.35):
        _sc, sim, frame, o, fi = best
        save_face_crop(frame, o, out_jpg)
        return dict(found=True, frame=fi, face_h=round(float(o.h), 1),
                    sim=(None if face_ref is None else round(sim, 3)),
                    note="人脸检测"), o.feat.astype(np.float32)

    # 兜底：全程没检出合格人脸，就裁目标最大那一帧的头部区域，保证文件存在
    fi, s = vis[0]
    frame = read_frame_at(video, fi)
    if frame is not None:
        save_head_crop(frame, (s.x1, s.y1, s.x2, s.y2), out_jpg)
        return dict(found=False, frame=fi, face_h=None, sim=None,
                    note="无可用人脸，已裁头部区域"), None
    return dict(found=False, frame=None, face_h=None, sim=None, note="未生成"), None


def save_face_crop(frame, obs, out_jpg: Path) -> None:
    import cv2

    w, h = obs.x2 - obs.x1, obs.y2 - obs.y1
    x1 = int(max(0, obs.x1 - 0.45 * w))
    x2 = int(min(frame.shape[1], obs.x2 + 0.45 * w))
    y1 = int(max(0, obs.y1 - 0.55 * h))
    y2 = int(min(frame.shape[0], obs.y2 + 0.45 * h))
    sub = frame[y1:y2, x1:x2]
    if sub.size == 0:
        return
    if min(sub.shape[:2]) < 224:
        f = 224.0 / max(1, min(sub.shape[:2]))
        sub = cv2.resize(sub, (int(sub.shape[1] * f), int(sub.shape[0] * f)),
                         interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(out_jpg), sub, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def save_head_crop(frame, box, out_jpg: Path) -> None:
    import cv2

    x1, y1, x2, y2 = [float(v) for v in box]
    h = y2 - y1
    w = x2 - x1
    X1 = int(max(0, x1 - 0.25 * w))
    X2 = int(min(frame.shape[1], x2 + 0.25 * w))
    Y1 = int(max(0, y1 - 0.10 * h))
    Y2 = int(min(frame.shape[0], y1 + 0.42 * h))
    sub = frame[Y1:Y2, X1:X2]
    if sub.size:
        cv2.imwrite(str(out_jpg), sub, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def _analyze_taq(video: Path, work: Path, args, seq_name: str = "") -> None:
    """用 TAQ-MOTR 出轨迹，替换 BoT-SORT。

    这就是做出 seg_002_taq_v3 那批结果的那条路：TAQ 在交叉/长时间遮挡下的
    轨迹传播比 BoT-SORT 强一个量级（seg_002 上空洞 1.5s vs 10s）。
    代价是慢——1080 代理上约 9 帧/秒，6383 帧要 12 分钟。
    """
    # 序列名必须跟着原片走：用了转码中间文件的话，video.stem 会变成 proxy_h264，
    # 于是抽帧重来、TAQ 结果也找不到（等于白跑一小时）。
    if not (ROOT / "TAQ-MOTR-main").exists():
        raise SystemExit(
            "\n--tracker taq 需要 TAQ-MOTR-main 目录（推理代码和权重），本项目未包含它。\n"
            "主流程请用默认的 anchor 模式（人脸锚点 + 快速检测），不需要 TAQ。\n")
    seq = seq_name or slugify(video.stem)
    mot = work / "taq" / f"{seq}.mot.txt"
    if args.redo_tracks or not mot.exists():
        cmd = [sys.executable, str(ROOT / "tools" / "taq_track.py"),
               "--video", str(video), "--seq", seq, "--out", str(mot),
               "--height", str(args.taq_height), "--tol", str(args.taq_tol),
               "--gpus", str(args.gpus)]
        if args.redo_tracks:
            cmd.append("--force")
        subprocess.run(cmd, check=True)
    else:
        print(f"[taq] 复用轨迹 {mot.name}")
    subprocess.run([sys.executable, str(ROOT / "tools" / "import_external_tracks.py"),
                    "--taq", str(mot), "--video", str(video), "--outdir", str(work),
                    "--single-shot"],
                   check=True)


def _track_fp(video: Path, args) -> dict:
    """决定"轨迹是否还能用"的全部输入。

    只判断 tracks.jsonl 存不存在是不够的——上一轮用 BoT-SORT 跑过，这一轮
    改 --tracker taq，文件还在，就会被当成有效缓存直接复用，参数写进 manifest
    却完全没生效。所以把影响轨迹的东西都记下来，变了就作废。
    """
    st = video.stat()
    return dict(tracker=str(args.tracker).lower(),
                taq_tol=(int(args.taq_tol) if str(args.tracker).lower() == "taq" else 0),
                taq_height=(int(args.taq_height) if str(args.tracker).lower() == "taq" else 0),
                # 快速检测的输入宽度/批大小会影响检测框，属于"轨迹层"的输入。
                # 之前漏了，改 --det-w 后 tracks.jsonl 还在就被当有效缓存复用，参数没生效。
                det_w=(int(args.det_w) if str(args.face_mode) == "anchor" else 0),
                det_batch=(int(args.det_batch) if str(args.face_mode) == "anchor" else 0),
                limit=(int(args.limit) if args.limit else 0),
                src_size=int(st.st_size), src_mtime=round(float(st.st_mtime), 1))


def _plan_fp(args) -> dict:
    """决定"取景与成片是否还能用"的输入（改了要重算路径和渲染）。"""
    return dict(zoom=float(args.zoom), preset=str(args.preset),
                max_targets=int(args.max_targets), targets=str(args.targets or ""),
                min_height=float(args.min_height), height_ratio=float(args.height_ratio),
                min_coverage=float(args.min_coverage),
                gid=int(args.gid), ref_face=str(args.ref_face or ""),
                face_check=float(args.face_check), face_mode=str(args.face_mode),
                face_thr=float(args.face_thr), face_every=int(args.face_every),
                face_topk=int(args.face_topk), face_up=int(args.face_up),
                anchor_gap=int(args.anchor_gap), anchor_grow=int(args.anchor_grow),
                csv_eps=float(args.eps), csv_tol=float(args.tol))


def _drop(work: Path, out_dir: Path, level: str) -> None:
    """作废缓存。level = tracks / identity / plan，逐级向下清。

    成片不删：旧的一版移到 out_dir/_prev/<时间戳>/ 存档，方便前后对比。
    中间产物（jsonl/json）体积大且不再使用，直接清掉。
    """
    prev = out_dir / "_prev" / time.strftime("%Y%m%d_%H%M%S")
    moved = False
    for d in sorted(out_dir.glob("ID*")):
        if d.is_dir() and any(d.iterdir()):
            prev.mkdir(parents=True, exist_ok=True)
            dst = prev / d.name
            n = 1
            while dst.exists():
                dst = prev / f"{d.name}_{n}"
                n += 1
            shutil.move(str(d), str(dst))
            moved = True
    mf = out_dir / "manifest.json"
    if mf.exists():
        prev.mkdir(parents=True, exist_ok=True)
        shutil.move(str(mf), str(prev / "manifest.json"))
        moved = True
    if moved:
        print(f"  上一版产物已存档：{prev.relative_to(out_dir)}（没有删除）")
    names = ["camera_path.json", "camera_stats.json", "quality_report.json",
             "target_summary.json"]
    if level in ("tracks", "identity"):
        names += ["tracks_kept.jsonl", "identity_events.json", "identity_faces.json",
                  "identity_stats.json", "identity_tracks.json", "local_tracks.json",
                  "target_tracks.jsonl"]
    if level == "tracks":
        names += ["tracks.jsonl", "shots.json", "trim_report.json", "analyze_stats.json"]
    for n in names:
        f = work / n
        if f.exists():
            f.unlink()


def _outputs_complete(out_dir: Path, manifest_p: Path) -> bool:
    """产物是否齐全：manifest 里每个 target 的 mp4/csv/jpg 三件都在。

    光有 manifest 不够——上一次可能渲染到一半就中断了，manifest 写了但成片没
    落盘。逐个核对文件存在性（>0 字节），缺一个都不算完成。
    """
    try:
        m = read_json(manifest_p)
    except (ValueError, OSError):
        return False
    targets = m.get("targets") or []
    if not targets:
        return False
    for t in targets:
        files = t.get("files") or {}
        for key in ("mp4", "csv", "jpg"):
            name = files.get(key)
            if not name:
                return False
            p = out_dir / t.get("id", "") / name
            if not p.exists() or p.stat().st_size == 0:
                return False
    return True


def _review_preview(video: str, boxes_by_fi: dict[int, list], out_jpg: Path,
                    n_shot: int = 6) -> Path | None:
    """低质量拦截时，抽几帧源画面拼一张预览，标出检测框，帮用户一眼看出问题。

    boxes_by_fi: {帧号: [(x1,y1,x2,y2), ...]}（源像素坐标）。抽 n_shot 帧，
    缩到 640 宽拼成一列，每帧把检测框画出来。用户看这张图就能判断"是不是
    检测框没框住目标，还是目标根本不在这一帧"。
    """
    import cv2

    import numpy as np
    from core.fastvideo import pipe_frames

    if not boxes_by_fi:
        return None
    fis = sorted(boxes_by_fi)
    step = max(1, len(fis) // n_shot)
    picks = fis[::step][:n_shot]
    tiles = []
    for fi, frame in pipe_frames(video, width=1280):
        if fi not in picks:
            continue
        img = frame
        h, w = img.shape[:2]
        for b in boxes_by_fi.get(fi, []):
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        t = fi / 60.0 if fi < 60 * 3600 else fi
        cv2.putText(img, f"fi={fi}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        tiles.append(img)
        if len(tiles) >= n_shot:
            break
    if not tiles:
        return None
    # 统一宽度拼成竖列
    tw = 640
    resized = [cv2.resize(t, (tw, int(t.shape[0] * tw / t.shape[1])),
                          interpolation=cv2.INTER_AREA) for t in tiles]
    sheet = np.vstack(resized)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        buf.tofile(str(out_jpg))
        return out_jpg
    return None



def _ensure_info(work: Path, video: Path) -> dict:
    """video_info.json 缺失就按源文件补写。

    检测要跑好几分钟，中间任何一次误删/中断都会让后面的 identify 直接崩在
    FileNotFoundError 上，而已经跑完的 tracks.jsonl 其实完好。这里自愈一下，
    不值得为一个小文件重跑整遍检测。
    """
    p = work / "video_info.json"
    if p.exists():
        return read_json(p)
    vi = probe(video)
    write_json(p, dict(path=str(Path(video).resolve()), width=vi.width, height=vi.height,
                       fps=vi.fps, n_frames=vi.n_frames, duration=vi.duration,
                       has_audio=vi.has_audio, codec=vi.codec))
    print(f"[warn] {work.parent.name}/video_info.json 缺失，已按源文件补写"
          f"（不影响已完成 {vi.n_frames} 帧的检测结果）")
    return read_json(p)


def _auto_ref_face(v: Path) -> Path | None:
    """找参考图：先认同名（SoCool.mp4 → SoCool.jpg），再认数字前缀（3-xxx.mp4 → 3.jpg）。

    命名两种都很常见，只认前缀会漏掉最常见的那种——视频和参考图放在一起、
    名字完全一样，本来是最自然的做法。
    """
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = v.parent / f"{v.stem}{ext}"
        if p.exists():
            return p
    m = re.match(r"(\d+)", v.stem)
    if m:
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            p = v.parent / f"{m.group(1)}{ext}"
            if p.exists():
                return p
    return None


def _grid_preview(mp4: Path, out_jpg: Path, cols: int = 4, rows: int = 3) -> Path | None:
    """从成片随机抽 cols*rows 帧拼成宫格图，供快速浏览整条成片。

    用 ffmpeg -ss 快速 seek 随机时间点，缩到每格 480 宽拼成宫格。抽的是**成片**
    （竖屏直拍），不是原片——用户要看的是最终交付的画面。
    """
    import cv2
    import random as _rnd

    import numpy as np
    from core.fastvideo import probe_size

    if not mp4.exists() or mp4.stat().st_size == 0:
        return None
    try:
        _, _, fps = probe_size(str(mp4))
        duration = mp4.stat().st_size  # 占位，下面用 ffprobe 拿真实时长
    except Exception:                                   # noqa: BLE001
        return None
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(mp4)], capture_output=True, text=True)
    try:
        duration = float(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None
    if duration <= 0:
        return None

    n = cols * rows
    # 随机抽 n 个时间点，避开首尾 2%（黑场/淡入淡出）
    lo, hi = duration * 0.02, duration * 0.98
    if hi <= lo:
        lo, hi = 0.0, max(duration - 0.05, 0.05)
    times = sorted(_rnd.sample(range(n * 40), n))
    times = [lo + (hi - lo) * (t / (n * 40)) for t in times]

    tiles = []
    for t in times:
        tmp = out_jpg.parent / "_grid_tmp.jpg"
        rr = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}",
                             "-i", str(mp4), "-frames:v", "1", "-q:v", "3", str(tmp)],
                            capture_output=True)
        if rr.returncode != 0 or not tmp.exists():
            continue
        buf = np.fromfile(str(tmp), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
        if img is None:
            continue
        tiles.append(img)
    tmp.unlink(missing_ok=True)
    if not tiles:
        return None

    cell_w = 480
    resized = [cv2.resize(im, (cell_w, int(im.shape[0] * cell_w / im.shape[1])),
                          interpolation=cv2.INTER_AREA) for im in tiles]
    cell_h = max(im.shape[0] for im in resized)
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 20, np.uint8)
    for i, im in enumerate(resized):
        if i >= n:
            break
        r, c = divmod(i, cols)
        h = im.shape[0]
        y0 = r * cell_h + (cell_h - h) // 2
        x0 = c * cell_w
        sheet[y0:y0 + h, x0:x0 + cell_w] = im
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(str(out_jpg))
        return out_jpg
    return None


def _preflight(src: Path) -> None:
    """启动预检：环境问题一次性说清，别跑几分钟才崩。

    检查 ffmpeg/ffprobe 是否在 PATH、检测权重是否存在、源目录里有没有"看起来像
    视频但格式不支持"的文件（比如 .webm 旧版没加支持，会静默漏掉）。
    """
    # 1) ffmpeg / ffprobe
    import shutil as _sh

    for exe in ("ffmpeg", "ffprobe"):
        if _sh.which(exe) is None:
            raise SystemExit(
                f"找不到 {exe}。请安装 FFmpeg 并加入 PATH：\n"
                f"  https://www.gyan.dev/ffmpeg/builds/  （下载 release，解压后把 bin 加进 PATH）")

    # 2) 检测权重
    weights = ROOT / "work" / "models" / "yolo11m.pt"
    if not weights.exists():
        raise SystemExit(f"找不到检测权重：{weights}\n"
                         f"  请确认 yolo11m.pt 在 work/models/ 下")

    # 3) 源目录里的非支持格式媒体文件（.webm 之外的老格式等），提示别静默漏掉
    known_media = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm",
                   ".ts", ".mts", ".m2ts", ".flv", ".wmv", ".3gp", ".rmvb"}
    unknown = []
    for p in src.iterdir():
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in known_media and suf not in VIDEO_EXT:
            unknown.append(p)
    if unknown:
        print(f"[!] {src} 里有 {len(unknown)} 个暂不支持的视频格式（会被跳过）：")
        for p in unknown:
            print(f"    - {p.name}（{p.suffix}）")


def _pick_person(v: Path, frames: int = 16) -> Path | None:
    """缺参考图时，自动从视频里挑人：抽帧 → 检测 → 人脸聚类 → 弹拼图 → 读编号。

    把 tools/extract_people.py 当成子进程跑（它已经是独立可用的命令行工具），
    拿到 faces/personNN.jpg 后：
      1. 弹系统看图器打开 preview.png（无损大图，最多 24 张候选）
      2. 命令行读一个编号（1..N），回车确认
      3. 把对应的 personNN.jpg 复制成 <视频名>.jpg 存到视频旁，run.py 自动认成参考图

    返回新参考图路径；用户按 Ctrl+C / 输入 q 放弃时返回 None。
    """
    tool = ROOT / "tools" / "extract_people.py"
    if not tool.exists():
        print(f"  [warn] 找不到挑人工具 {tool}，跳过自动选人")
        return None
    # 显式指定输出目录，别让 extract_people 自己推断——两个模块的 slug 规则、
    # 路径解析可能不一致，推断错就找不到 faces/。
    people_dir = v.with_name(f"{slugify(v.stem)}_people")
    print(f"\n  没有参考图，自动从视频里挑人……")
    r = subprocess.run([sys.executable, str(tool), "--video", str(v),
                        "--out", str(people_dir), "--frames", str(frames)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="ignore")
    if r.returncode != 0 or not (people_dir / "faces").exists():
        err = (r.stderr or "").strip().splitlines()
        print(f"  [warn] 挑人失败（rc={r.returncode}）："
              + (err[-1][:120] if err else "未产出 faces/"))
        print(f"  请手动指定 --ref-face")
        return None

    faces = sorted((people_dir / "faces").glob("person*.jpg"))
    if not faces:
        print(f"  [warn] 没检出可用人脸，请手动指定 --ref-face")
        return None

    # 弹拼图给用户看（PNG 无损大图，选人要看清脸）
    preview = people_dir / "preview.png"
    if not preview.exists():
        preview = people_dir / "preview.jpg"     # 兼容旧版
    if preview.exists():
        try:
            os.startfile(str(preview))           # noqa: S606  打开系统默认看图器
        except OSError:
            print(f"  拼图已生成：{preview}")

    print(f"\n  从视频里检出 {len(faces)} 个候选（见弹出来的 preview.png 拼图）")
    print(f"  输入编号选一个当目标（回车默认 1，q 跳过）：", end=" ", flush=True)
    try:
        ans = sys.stdin.readline().strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not ans:
        ans = "1"
    if ans.lower() in ("q", "quit", "n", "no", "skip"):
        return None
    if not ans.isdigit():
        print(f"  无效输入 '{ans}'，跳过")
        return None
    k = int(ans)
    if k < 1 or k > len(faces):
        print(f"  编号 {k} 超出范围 1..{len(faces)}，跳过")
        return None

    src_face = faces[k - 1]
    dst_face = v.with_name(f"{v.stem}.jpg")
    try:
        shutil.copyfile(src_face, dst_face)
    except OSError as exc:
        print(f"  [warn] 复制参考图失败：{exc}")
        return None
    print(f"  已选 person{k:02d}，存成 {dst_face.name}，继续处理")
    return dst_face


def _ensure_proxy(v: Path, work: Path) -> Path:
    """AV1 这类 OpenCV 硬解拿不到帧、软解又只有 8 FPS 的格式，先转一份 H.264。

    不转的代价是实打实的：23070 帧的 4K AV1，OpenCV 软解 8 FPS → 单次解码 48
    分钟，而 analyze 的镜头检测那一步还要再解一遍，渲染再解一遍。转一次约 15
    分钟，之后每一步都吃到硬件加速（约 54 FPS）。crf 16 近无损，肉眼无差。
    """
    try:
        vi = probe(v)
    except Exception:                                  # noqa: BLE001
        return v
    codec = str(getattr(vi, "codec", "") or "").lower()
    if "av1" not in codec:
        return v
    proxy = work / "proxy_h264.mp4"
    if proxy.exists() and proxy.stat().st_size > 0 \
            and proxy.stat().st_mtime > v.stat().st_mtime:
        print(f"[{work.parent.name}] 复用中间转码 {proxy.name}")
        return proxy
    print(f"[{work.parent.name}] 源编码 {codec.upper()}：OpenCV 软解仅约 8 FPS，"
          f"先转一份 H.264（近无损）...")
    t = time.time()
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(v),
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
                        "-pix_fmt", "yuv420p", "-c:a", "copy", str(proxy)])
    if r.returncode != 0 or not proxy.exists() or proxy.stat().st_size == 0:
        print(f"[{work.parent.name}] 转码失败，直接用原片（会慢很多倍）")
        return v
    print(f"[{work.parent.name}] 转码完成 {time.time() - t:.0f}s "
          f"（{proxy.stat().st_size / 1e9:.1f}GB）")
    return proxy


# ---------------------------------------------------------------- 单个视频
def process_video(v: Path, args, out_root: Path) -> dict:
    slug = slugify(v.stem)
    prefix = args.prefix or slug
    out_dir = out_root / slug
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    cfg = cfgmod.load(str(ROOT / "configs/default.yaml"))
    src_v = v                       # 原片：名字、音轨、交付说明都用它

    # 缓存是否还算数，看指纹而不是"文件在不在"。改了 --tracker / --preset
    # 之类的参数，旧产物必须作废，否则跑出来和上一轮一模一样还以为没生效。
    fp_path = work / "run_fingerprint.json"
    fp_old = read_json(fp_path) if fp_path.exists() else None

    # 参考图的优先级：命令行显式指定 > 视频旁边的同名图 > 上次锁定的那位。
    # "同名图"必须排在"沿用上次"前面——否则批量跑时，第一条视频的参考图会被
    # 装进 args 带到后面所有视频（实测 wuface 用了 SoCool.jpg，锚点全 0）。
    # 注意：这段必须放在转码判断之前——有参考图时走 anchor（ffmpeg 管道直解），
    # 没参考图才走 OpenCV 老链路转码；顺序反了会让有参考图的 AV1 也白转一次码。
    auto_ref = _auto_ref_face(src_v)
    if auto_ref is not None and not args.ref_face:
        args.ref_face = str(auto_ref)
        print(f"[{slug}] 参考图：{auto_ref.name}")
    elif fp_old and not args.gid and not args.ref_face:
        pg = int(fp_old.get("plan", {}).get("gid", 0) or 0)
        pr = str(fp_old.get("plan", {}).get("ref_face", "") or "")
        if pg or pr:
            args.gid, args.ref_face = pg, pr
            print(f"[{slug}] 沿用上次锁定："
                  + (f"gid{pg}" if pg else f"参考图 {Path(pr).name}"))

    # anchor 模式走 ffmpeg 管道（能直接解 AV1，4K 37.7 FPS / 缩到 960 后上百 FPS），
    # 不再需要预先转码；只有走 OpenCV 的老链路才转。
    if not (str(args.face_mode) == "anchor" and args.ref_face):
        v = _ensure_proxy(v, work)
    cfg["input"]["video"] = str(v)
    cfg["input"]["limit"] = args.limit
    cfg["output"]["dir"] = str(work)

    t_all = time.time()
    kept_p = work / "tracks_kept.jsonl"

    fp_new = dict(tracks=_track_fp(src_v, args), plan=_plan_fp(args))

    # anchor 模式：只用人脸锚点 + 检测框，不需要身份聚类。
    # 这条路上 identify（人体 ReID + 全局聚类）完全是白跑的——4K 长片上要半小时
    # 到一小时，而它产出的那一堆 global id 我们一个都不用。
    anchor_mode = (str(args.face_mode) == "anchor" and bool(args.ref_face))

    need_tracks = bool(args.redo_all or args.redo_tracks or fp_old is None
                       or fp_old.get("tracks") != fp_new["tracks"]
                       or not ((work / "tracks.jsonl").exists()
                               or (anchor_mode and kept_p.exists()))
                       or (not anchor_mode and not kept_p.exists()))
    need_plan = bool(need_tracks or fp_old is None
                     or fp_old.get("plan") != fp_new["plan"])

    # —— 无变化直接跳过：指纹一致、产物齐全、也没要求重跑，就整段跳过。
    # 否则"参数未变，复用分析结果"之后仍会重新规划/导出/渲染，重复等旧视频。
    if not need_tracks and not need_plan and not args.redo_all:
        manifest_p = out_dir / "manifest.json"
        if manifest_p.exists() and _outputs_complete(out_dir, manifest_p):
            print(f"[{slug}] 跳过：参数与素材未变，产物齐全（{out_dir.name}）")
            return dict(slug=slug, video=str(src_v), status="skipped",
                        targets=read_json(manifest_p).get("targets", []))

    if need_tracks:
        if fp_old is None:
            print(f"[{slug}] 无参数指纹（旧目录），按本次参数重跑全流程")
        elif args.redo_all:
            print(f"[{slug}] --redo-all，重跑全流程")
        elif args.redo_tracks:
            print(f"[{slug}] --redo-tracks，重跑跟踪")
        else:
            diff = ", ".join(f"{k}: {fp_old.get('tracks', {}).get(k)} → {v2}"
                             for k, v2 in fp_new["tracks"].items()
                             if fp_old.get("tracks", {}).get(k) != v2)
            print(f"[{slug}] 跟踪参数变化（{diff}），作废旧轨迹重跑")
        _drop(work, out_dir, "tracks")
    elif need_plan:
        print(f"[{slug}] 取景参数变化，只重算路径与渲染")
        _drop(work, out_dir, "plan")
    else:
        print(f"[{slug}] 参数未变，复用已有分析结果")

    if need_tracks:
        try:
            t = time.time()
            if anchor_mode:
                # anchor 只要每帧的框，不需要轨迹 ID。走 ffmpeg 管道 + 批量 YOLO，
                # 跳过跟踪器和镜头检测，实测 4K AV1 上从 4 FPS 提到 84 FPS。
                print(f"[{slug}] analyze（快速检测，无跟踪）...")
                detect_fast.run(str(v), str(work), det_w=int(args.det_w),
                                batch=int(args.det_batch),
                                weights=str(cfg["detection"]["weights"]),
                                limit=int(args.limit or 0))
            elif str(args.tracker).lower() == "taq":
                print(f"[{slug}] analyze（TAQ-MOTR）...")
                _analyze_taq(v, work, args, seq_name=slug)
            else:
                print(f"[{slug}] analyze ...")
                analyze.run(cfg, force=args.redo_all)
            print(f"[{slug}] analyze {time.time() - t:.0f}s")
            if anchor_mode:
                print(f"[{slug}] 跳过身份聚类（anchor 模式只需检测框）")
            else:
                print(f"[{slug}] identify ...")
                t = time.time()
                identify.run(cfg)
                print(f"[{slug}] identify {time.time() - t:.0f}s")
        except (SystemExit, Exception) as exc:      # noqa: BLE001
            if not kept_p.exists():
                raise
            print(f"[warn] {slug} 选人阶段失败（{exc}），用已有轨迹自己选人")

    # 轨迹这一层走到这里就是可用的，立刻记账；plan 那层等渲染完再记。
    meta = read_json(fp_path) if fp_path.exists() else {}
    meta["tracks"] = fp_new["tracks"]
    write_json(fp_path, meta)

    info = _ensure_info(work, v)
    info["source_name"] = src_v.name
    if v != src_v:
        info["path"] = str(v)
        info["origin"] = str(src_v)
    fps = float(info["fps"])
    n_frames = int(info["n_frames"])
    if args.limit:
        n_frames = min(n_frames, args.limit)

    if anchor_mode:
        global _REF_FEAT
        _REF_FEAT = _ref_feature(str(args.ref_face))
        pool_p = work / "tracks.jsonl"
        if not pool_p.exists():
            pool_p = kept_p
        rows = list(read_jsonl(pool_p))
        tracks = manager.build_local_tracks(rows)
        for r in rows:
            t2 = tracks.get((int(r["sid"]), int(r["lid"])))
            if t2 is not None:
                t2.global_id = int(r.get("gid", r.get("lid", -1)))
        ident = []
        cands = [dict(gid=-1, rank=1, cov=1.0, h_p90=0.0, merge_gids=[],
                      score=1.0, n=len(tracks), frames=n_frames)]
        dropped = []
        print(f"[{slug}] anchor 模式：参考人脸定位，检测框 {len(rows)} 个 / "
              f"{len(tracks)} 条局部轨迹")
    else:
        tracks, ident, _by_gid = load_identities(work)
        cands, dropped = pick_candidates(ident, tracks, info, n_frames, args)
        cands = _lock_target(cands, work / "identity_faces.json", args)
    print(f"[{slug}] 候选 {len(cands)} 个：" + ", ".join(
        f"ID{r['rank']}(gid{r['gid']}, 覆盖{r['cov'] * 100:.0f}%, h{r['h_p90']:.0f})"
        for r in cands[:8]))
    for r in dropped[:6]:
        print(f"        丢弃 gid{r['gid']}: {r.get('reason', '')} "
              f"(覆盖率 {r['cov'] * 100:.0f}%, 框高 {r['h_p90']:.0f})")

    if not cands:
        return dict(slug=slug, video=str(v), status="review",
                    reason="无合格候选", targets=[])

    picked = cands[: args.max_targets]
    if args.targets:
        want = []
        for tok in str(args.targets).replace("，", ",").split(","):
            tok = tok.strip().lstrip("IDid")
            if not tok.isdigit():
                continue
            k = int(tok) - 1
            if 0 <= k < len(cands):
                want.append(cands[k])
            else:
                print(f"[warn] --targets {tok} 超出候选数 {len(cands)}")
        if want:
            picked = want

    # 上次跑出的 ID 目录如果这次没被选中，会留在原地造成误读，明确提示一下
    stale = sorted(p.name for p in out_dir.glob("ID*")
                   if p.is_dir() and p.name not in {f"ID{r['rank']}" for r in picked})
    if stale:
        print(f"[{slug}] 注意：{out_dir.name} 里还有上次的 {', '.join(stale)}，"
              f"本次没重做；不需要就手动删掉")

    out_w = int(cfg["export"]["width"])
    out_h = int(cfg["export"]["height"])
    ccfg = dict(cfg["camera"])
    ccfg.update(zoom=float(args.zoom), preset=str(args.preset), crop_width=0,
                out_width=out_w)
    body_min = float(cfg["identity"].get("body_min_ref", 0.64))

    # ---- 规划取景
    jobs, recs = [], []
    for r in picked:
        gid = r["gid"]
        if anchor_mode:
            series, summary = [None] * n_frames, dict(coverage=0.0, n=0, segments=0)
        else:
            series, summary = manager.build_target_series(
                ident, tracks, gid, n_frames, fps, body_min=body_min,
                merge_gids=r.get("merge_gids") or [])

        if _REF_FEAT is not None and args.face_mode == "anchor":
            # 人脸锚点 + 局部框关联：不信任任何长轨迹，只信"脸转过来了"和
            # 转身后背对那 1.5 秒的框连续性。多人素材上这是唯一稳的路子。
            anch = face_anchor.scan_cached(
                info["path"], tracks, _REF_FEAT,
                work / f"face_scan_{face_anchor.ref_hash(_REF_FEAT)}.json",
                every=int(args.face_every), up=int(args.face_up),
                top_k=int(args.face_topk))
            series, ast = face_anchor.build_series(
                tracks, n_frames, fps, anch,
                thr=float(args.face_thr), max_gap=int(args.anchor_gap),
                max_grow=int(args.anchor_grow), gid=gid)
            st = face_anchor.summarize(series, fps)
            _hs = [x.y2 - x.y1 for x in series if x is not None]
            summary = dict(coverage=round(st["visible"], 4),
                           longest_gap_s=st["longest_gap_s"],
                           median_h=round(float(np.median(_hs)), 1) if _hs else 0.0,
                           n=sum(1 for x in series if x is not None), segments=0,
                           hit_ratio=round(ast.get("hit_ratio", 0.0), 4),
                           anchors=ast.get("anchors", 0), hits=ast.get("hits", 0))
            print(f"  ID{r['rank']} 人脸锚点：{ast['hits']}/{ast['anchors']} 个锚点过阈值"
                  f"（{ast['hit_ratio'] * 100:.0f}%，平均相似度 {ast['mean_sim']}），"
                  f"框关联补满后可见 {st['visible'] * 100:.1f}%，"
                  f"最长空洞 {st['longest_gap_s']}s")
        elif _REF_FEAT is not None and args.face_mode == "track":
            # 人脸直接驱动：绕开聚类，逐帧在检测框里找最像参考图的那一个
            series, fst = _series_by_face(
                tracks, info["path"], n_frames, fps, _REF_FEAT,
                every=int(args.face_every), thr=float(args.face_thr), gid=gid)
            print(f"  ID{r['rank']} 人脸驱动：命中 {fst['hits']} 帧，补洞后可见 "
                  f"{fst['ratio'] * 100:.1f}%，平均相似度 {fst['mean_sim']}")
        elif _REF_FEAT is not None:
            vst = _verify_series_face(series, info["path"], _REF_FEAT, thr=float(args.face_check))
            left = sum(1 for s in series if s is not None)
            print(f"  ID{r['rank']} 人脸核验：抽 {vst['checked']} 帧，剔除 {vst['rejected']} 处"
                  f"（剩余可见 {left / max(1, n_frames) * 100:.1f}%）")
        poses, cam = planner.plan(series, info, ccfg, gid)
        d = out_dir / f"ID{r['rank']}"
        d.mkdir(parents=True, exist_ok=True)
        base = d / f"{prefix}-ID{r['rank']}"
        jobs.append(dict(out=str(base.with_suffix(".mp4")), poses=poses))
        recs.append(dict(cand=r, gid=gid, base=base, poses=poses, series=series,
                         summary=summary, cam=cam))

    # ---- 人脸图（独立小规模解码，只取几个人选帧，不占渲染时间）
    faces_ref = {}
    fp = work / "identity_faces.json"
    if fp.exists():
        faces_ref = read_json(fp)
    for rec in recs:
        ref = faces_ref.get(str(rec["gid"]))
        ref = np.array(ref, dtype=np.float32) if ref else None
        rec["face"], rec["feat"] = export_face(info["path"], rec["series"],
                                              rec["base"].with_suffix(".jpg"), ref)

    # 自检：两个 ID 的人脸高度相似，说明身份被拆碎了或者选到了同一个人。
    # 不自动纠正（自动纠正会把"两个长得很像的舞者"误并成一个人），只报出来。
    dup = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i].get("feat"), recs[j].get("feat")
            if a is None or b is None:
                continue
            sim = float(a @ b)
            if sim >= 0.62:
                dup.append(dict(a=f"ID{recs[i]['cand']['rank']}",
                                b=f"ID{recs[j]['cand']['rank']}",
                                sim=round(sim, 3)))
    for w in dup:
        print(f"  [warn] {w['a']} 与 {w['b']} 人脸相似度 {w['sim']}，可能是同一个人")

    # ---- 点位 CSV（用最终 poses，不是原始检测点）
    for rec in recs:
        rec["points"] = pts.write_points(
            rec["base"].with_suffix(".csv"), rec["poses"], info,
            label=f"ID{rec['cand']['rank']}", gid=rec["gid"], out_w=out_w, out_h=out_h,
            eps=float(args.eps), tol_out=float(args.tol),
            extra=dict(coverage=rec["summary"]["coverage"],
                       median_h=round(float(rec["summary"].get("median_h")
                                            or rec["cand"]["h_p90"]), 1),
                       face_sim=rec["face"].get("sim") if rec.get("face") else None))

    # ---- 质量体检：零命中 / 覆盖极低只提醒，不再拦截渲染。
    # 以前全部 ID 不合格时这里会直接返回"需复核"、什么都不出片；但覆盖低不等于没用
    # （片段本来就短、参考脸只有侧脸、只想要某一小段都可能是这种情况），出不出片
    # 交给调用方判断，这里只把问题喊出来。
    # 判据：覆盖 < 15%（和 min_coverage 对齐）或锚点命中率 < 20%（=选错人）。
    quality_notes: list[str] = []
    low_quality = []
    for rec in recs:
        sm = rec["summary"]
        cov = sm.get("coverage", 0.0)
        hit = sm.get("hit_ratio")
        # hit_ratio 只在 anchor 模式存在；其它模式拿不到就只判 coverage
        bad_hit = hit is not None and hit < 0.20
        if cov < 0.15 or bad_hit:
            low_quality.append(rec)
    if low_quality:
        for rec in low_quality:
            sm = rec["summary"]
            quality_notes.append(
                f"ID{rec['cand']['rank']}(覆盖 {sm['coverage'] * 100:.1f}%"
                + (f", 锚点命中 {sm['hit_ratio'] * 100:.0f}%" if sm.get("hit_ratio") is not None else "")
                + f", 最长空洞 {sm.get('longest_gap_s', 0):.1f}s)")
        print(f"[{slug}] [warn] 质量偏低，照常渲染（" + "；".join(quality_notes) + "）")
        print(f"        可能是参考脸选错了，成片不对就换个参考图或检查 --ref-face")
        # 附一张抽帧预览：把检测框画在源帧上，用户一眼看出框没框住谁
        try:
            byf: dict[int, list] = {}
            for rr in rows:
                byf.setdefault(int(rr["fi"]), []).append(
                    (rr["x1"], rr["y1"], rr["x2"], rr["y2"]))
            pv = _review_preview(info.get("origin") or info["path"], byf,
                                 out_dir / "_review_preview.jpg")
            if pv is not None:
                print(f"        复核预览：{pv}")
        except Exception:                                # noqa: BLE001
            pass

    # ---- 一次解码、多路渲染
    # 命令行显式给了就用命令行的，否则回落到配置：HEVC 版 bat 靠这里切编码器。
    enc_codec = str(args.codec or cfg["export"].get("codec", "h264_nvenc"))
    enc_quality = int(args.quality if args.quality is not None
                      else cfg["export"].get("quality", 19))
    if jobs:
        print(f"[{slug}] render × {len(jobs)}  codec={enc_codec} cq={enc_quality}")
        t = time.time()
        enc = dict(codec=enc_codec, quality=enc_quality,
                   audio=bool(cfg["export"].get("preserve_audio", True)))
        if len(jobs) == 1 and str(args.render_engine) == "ffmpeg":
            # 单路时让 ffmpeg 自己逐帧裁剪：172 FPS，是 Python 路径的 5 倍。
            # 多路仍走 render_multi——一次解码分给多个编码器，Python 侧反而更省。
            try:
                res = [render.render_ffmpeg(info["path"], jobs[0]["poses"], jobs[0]["out"],
                                            fps, out_w=out_w, out_h=out_h, **enc)]
            except Exception as exc:                    # noqa: BLE001
                print(f"  ffmpeg 内部裁剪失败（{exc}），回退 Python 路径")
                res = render.render_multi(info["path"], jobs, fps, out_w=out_w,
                                          out_h=out_h, **enc)
        else:
            res = render.render_multi(info["path"], jobs, fps, out_w=out_w, out_h=out_h,
                                      **enc)
        print(f"[{slug}] render {time.time() - t:.0f}s")
    else:
        res = []

    # 成片都落盘了，取景这一层才算可用
    meta = read_json(fp_path) if fp_path.exists() else {}
    meta["plan"] = fp_new["plan"]
    write_json(fp_path, meta)

    # ---- 成片 12 宫格拼接图（快速浏览整条成片）
    for rec in recs:
        mp4_p = rec["base"].with_suffix(".mp4")
        if not mp4_p.exists() or mp4_p.stat().st_size == 0:
            continue
        grid_p = rec["base"].with_suffix(".grid.jpg")
        try:
            g = _grid_preview(mp4_p, grid_p)
            if g is not None:
                rec["grid"] = grid_p.name
        except Exception:                                # noqa: BLE001
            pass

    # ---- 汇总
    print(f"\n[{slug}] 结果")
    targets = []
    for rec, r in zip(recs, res or [None] * len(recs)):
        cam, sm, pc = rec["cam"], rec["summary"], rec["points"]
        line = (f"  ID{rec['cand']['rank']}  gid{rec['gid']}  "
                f"覆盖 {sm['coverage'] * 100:5.1f}%  全身 {cam['target_inside_ratio'] * 100:5.1f}%  "
                f"zoom {float(args.zoom):.2f}  关键帧 {pc['keyframes']}/{pc['frames']}  "
                f"曲线误差 {pc['max_error_px']:.2f}px  反向 {cam['reversals']}"
                + (f"  {r['size_mb']}MB" if r else ""))
        print(line)
        if not rec["face"].get("found"):
            print(f"        注意：{rec['face'].get('note')}")
        targets.append(dict(
            id=f"ID{rec['cand']['rank']}", gid=rec["gid"],
            coverage=sm["coverage"], full_body=round(cam["target_inside_ratio"], 4),
            head_inside=round(cam["head_inside_ratio"], 4),
            foot_inside=round(cam["foot_inside_ratio"], 4),
            reversals=cam["reversals"], crop=f"{pc['crop_w']}x{pc['crop_h']}",
            upscale=round(out_w / max(1, pc["crop_w"]), 3),
            keyframes=pc["keyframes"], frames=pc["frames"],
            frames_per_keyframe=pc["frames_per_keyframe"],
            curve_error_px=pc["max_error_px"], zoom=float(args.zoom),
            longest_gap_s=sm["longest_gap_s"],
            face=rec["face"],
            size_mb=(r["size_mb"] if r else None),
            files=dict(jpg=rec["base"].with_suffix(".jpg").name,
                       mp4=rec["base"].with_suffix(".mp4").name,
                       csv=rec["base"].with_suffix(".csv").name,
                       grid=rec.get("grid"))))

    manifest = dict(
        video=dict(path=str(v), width=info["width"], height=info["height"],
                   fps=info["fps"], n_frames=n_frames, duration=info["duration"]),
        slug=slug, prefix=prefix, out=dict(width=out_w, height=out_h),
        params=dict(tracker=str(args.tracker), gid=int(args.gid),
                    ref_face=str(args.ref_face or ""),
                    taq_tol=(int(args.taq_tol) if str(args.tracker) == "taq" else None),
                    taq_height=(int(args.taq_height) if str(args.tracker) == "taq" else None),
                    zoom=float(args.zoom), preset=str(args.preset),
                    max_targets=args.max_targets, min_height=args.min_height,
                    height_ratio=args.height_ratio, min_coverage=args.min_coverage,
                    csv_eps=float(args.eps), csv_tol=float(args.tol),
                    codec=enc_codec, cq=enc_quality),
        seconds=round(time.time() - t_all, 1),
        status="ok",
        # 质量体检只做提示，不影响出片；为空表示覆盖/命中都正常
        quality_notes=quality_notes,
        targets=targets, face_duplicates=dup,
        dropped=[dict(gid=r["gid"], reason=r.get("reason", ""),
                      coverage=r["cov"], h_p90=round(r["h_p90"], 1),
                      t0=r["t0"], t1=r["t1"]) for r in dropped])
    write_json(out_dir / "manifest.json", manifest)
    # 清理选人临时图：<视频名>_people/（preview.png + faces/ + frame_* 等）在选人
    # 时就完成了使命，参考图已经复制成 <视频名>.jpg，这里删掉大图释放空间。
    people_dir = src_v.with_name(f"{slug}_people")
    if people_dir.exists():
        shutil.rmtree(people_dir, ignore_errors=True)
    print(f"[{slug}] 用时 {time.time() - t_all:.0f}s → {out_dir}")
    return manifest


class _Tee:
    """把 stdout 同时写进日志文件。

    这样 bat 就不需要用外部管道做 tee——那条管道会把 Python 的退出码换成
    管道末端的退出码，失败时 bat 也以为成功了。
    """

    def __init__(self, stream, path: Path):
        self.stream = stream
        self.f = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, data):
        self.stream.write(data)
        try:
            self.f.write(data)
        except (ValueError, OSError):
            pass

    def flush(self):
        try:
            self.stream.flush()
            self.f.flush()
        except (ValueError, OSError):
            pass

    def isatty(self) -> bool:
        return False


# ---------------------------------------------------------------- 入口
def _resolve(p: str, must_exist: bool) -> Path:
    """相对路径：先当前目录，再项目根目录。"""
    q = Path(p).expanduser()
    if q.is_absolute():
        return q
    cwd = (Path.cwd() / q)
    root = (ROOT / q)
    if cwd.exists():
        return cwd.resolve()
    if root.exists():
        return root.resolve()
    return (cwd if must_exist else root)


def main():
    ap = argparse.ArgumentParser(description="VVERTIGO 一键直拍（目录批处理）")
    ap.add_argument("--src", default="待处理", help="待处理视频目录")
    ap.add_argument("--out", default="outputs", help="输出根目录")
    ap.add_argument("--max-targets", type=int, default=3, help="每个视频最多出几个 ID")
    ap.add_argument("--targets", default="", help="只出排名第几个，如 1,3")
    ap.add_argument("--gid", type=int, default=0,
                    help="锁定内部全局 ID（日志里的 gidN）；锁定后它就是 ID1")
    ap.add_argument("--ref-face", default="",
                    help="给一张该人的人脸图，自动锁定最像的那个 ID；它就是 ID1")
    ap.add_argument("--face-check", type=float, default=0.50,
                    help="参考人脸核验阈值：抽检帧的人脸相似度低于它就从目标里剔掉；0=关闭")
    ap.add_argument("--face-mode", default="anchor",
                    choices=["anchor", "track", "check"],
                    help="anchor=人脸锚点+局部框关联（多人/串人素材唯一稳的路子）；"
                         "track=逐帧人脸最佳匹配；check=只做事后核验")
    ap.add_argument("--anchor-gap", type=int, default=150,
                    help="两个锚点相隔超过这么多帧就不桥接（留空让相机 HOLD）")
    ap.add_argument("--render-engine", default="ffmpeg", choices=["ffmpeg", "python"],
                    help="ffmpeg=让 ffmpeg 逐帧裁剪（快 5 倍，时间戳对齐到帧）；"
                         "python=逐帧在 Python 里裁（像素级精确，慢 5 倍）")
    ap.add_argument("--codec", default=None,
                    choices=["h264_nvenc", "hevc_nvenc", "libx264"],
                    help="成片编码器，默认读 configs/default.yaml 的 export.codec "
                         "(h264_nvenc)。hevc_nvenc=H.265 硬编，同画质体积小 30-50%%")
    ap.add_argument("--quality", type=int, default=None,
                    help="质量档（NVENC 的 cq / x264 的 crf），默认读配置，19 左右")
    ap.add_argument("--face-gpu", action="store_true",
                    help="SFace 走 GPU；实测和 torch 抢卡后反而慢 5 倍，默认关")
    ap.add_argument("--face-topk", type=int, default=10,
                    help="每帧检查几个最大的框；群舞素材要够大，否则漏掉目标")
    ap.add_argument("--face-up", type=int, default=2,
                    help="头部裁出后放大倍数；1 快 3 倍但相似度略降（实测 0.672 vs 0.689）")
    ap.add_argument("--det-w", type=int, default=1280,
                    help="快速检测的输入宽度：1280 约 46 FPS 且框精度与旧链路一致；"
                         "960 约 82 FPS 但覆盖会掉约 9 个点；0=用源分辨率（4K 仅 4 FPS，别用）")
    ap.add_argument("--det-batch", type=int, default=8,
                    help="检测批大小")
    ap.add_argument("--anchor-grow", type=int, default=60,
                    help="单次框关联最多补多少帧，防止靠惯性漂走")
    ap.add_argument("--face-thr", type=float, default=0.55,
                    help="人脸驱动时判定'这是她'的相似度阈值")
    ap.add_argument("--face-every", type=int, default=8,
                    help="人脸驱动的采样间隔（帧）；越小越准也越慢")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="1.0 = 9:16 最大裁切（最清晰）；>1 收紧构图并放大")
    ap.add_argument("--preset", default="smooth", choices=["follow", "smooth", "steady"])
    ap.add_argument("--prefix", default="", help="文件名前缀，默认用视频名")
    ap.add_argument("--min-height", type=float, default=0.20,
                    help="硬门槛：框高低于画面高的该比例视为观众/背景")
    ap.add_argument("--height-ratio", type=float, default=0.55,
                    help="相对门槛：框高低于最高者的该比例视为次要人物")
    ap.add_argument("--min-coverage", type=float, default=0.15,
                    help="至少覆盖全片多少比例才算候选")
    ap.add_argument("--eps", type=float, default=3.0,
                    help="CSV 稀疏化容差（源像素）；越大关键帧越少")
    ap.add_argument("--tol", type=float, default=1.5,
                    help="导入 AE 后允许的曲线误差（成片像素）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 帧（调试）")
    ap.add_argument("--only", default="", help="只处理文件名包含该串的视频")
    ap.add_argument("--redo-all", action="store_true", help="忽略缓存，连检测一起重跑")
    ap.add_argument("--redo-tracks", action="store_true",
                    help="只重跑跟踪（TAQ 会重新抽帧+推理，很慢）")
    ap.add_argument("--no-open", action="store_true",
                    help="完成后不自动打开结果目录（无人值守/脚本调用时用）")
    ap.add_argument("--tracker", default="auto", choices=["auto", "taq", "botsort"],
                    help="跟踪器：auto=给了参考人脸就走 botsort（人脸锚点只要检测框，"
                         "快 6-8 倍且框更准），否则 taq（v3 用的就是它）；"
                         "也可强制 taq / botsort")
    ap.add_argument("--taq-tol", type=int, default=600, choices=[30, 600],
                    help="TAQ 丢失容忍帧数，600 = 放宽到 10 秒")
    ap.add_argument("--taq-height", type=int, default=1080,
                    help="TAQ 抽帧高度（代理分辨率，框会按比例还原回源坐标）")
    ap.add_argument("--gpus", default="0")
    args = ap.parse_args()
    try:
        sys.stdout = _Tee(sys.stdout, ROOT / "run_log.txt")
    except OSError:
        pass
    if getattr(args, "face_gpu", False):
        os.environ["VVERTIGO_SFACE_GPU"] = "1"

    # 相对路径先按当前目录解析，找不到再退回项目根目录。
    # 这样 `cd 待处理; py -3.12 ..\run.py --src .` 和 `py -3.12 run.py --src 待处理` 都能跑。
    src = _resolve(args.src, must_exist=True)
    out_root = _resolve(args.out, must_exist=False)
    if not src.exists():
        raise SystemExit(f"源目录不存在: {src}\n"
                         f"  当前目录: {Path.cwd()}\n"
                         f"  项目根目录: {ROOT}\n"
                         f"  用法: py -3.12 run.py --src 待处理")

    # 参考图先校验，别等清完缓存、跑完取景才在这里失败
    if args.ref_face:
        q = _resolve(str(args.ref_face), must_exist=False)
        if not q.exists():
            raise SystemExit(f"--ref-face 读不到图片：{q}\n"
                             f"  提示：outputs/<视频>/IDn/ 下的人脸 jpg 可以直接拿来用")
        args.ref_face = str(q)

    # 启动预检：ffmpeg / 权重 / 不支持的格式，一次说清
    _preflight(src)

    vids = sorted([p for p in src.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_EXT],
                  key=lambda p: p.name)
    if args.only:
        vids = [v for v in vids if args.only in v.name]
    if not vids:
        raise SystemExit(f"{src} 下没有视频文件（支持 {', '.join(VIDEO_EXT)}）")

    print(f"待处理 {len(vids)} 个视频 → {out_root}")
    for v in vids:
        print(f"  - {v.name}")

    # 没有参考图时的处理：逐个进入"看图选人"，选完直接处理。
    # 有 --ref-face 时跳过；能自动匹配到同名/同前缀图的也跳过。
    if str(args.face_mode) == "anchor":
        picked_ref = {}
        for v in vids:
            if _auto_ref_face(v) is not None:
                continue                    # 已有参考图，直接处理
            print(f"\n{'=' * 68}")
            face = _pick_person(v)
            if face is not None:
                picked_ref[str(v)] = str(face)
            else:
                print(f"  [warn] {v.name} 未选人，本次跳过（之后可再指定 --ref-face）")
                vids = [x for x in vids if x != v]
        # 把选好的参考图注入 args，process_video 里 _auto_ref_face 会再次命中
        # （此时同名图已生成），所以这里其实只需保证文件在；但显式记一笔更清楚。
        if picked_ref:
            print(f"\n已为 {len(picked_ref)} 个视频选定参考图")

    if str(args.tracker) == "auto":
        # 有 TAQ 目录、且没给参考图（走旧链路的聚类选人）时才用 TAQ。
        # 本项目不带 TAQ-MOTR，所以默认落到 botsort，不会再撞上"目录不存在"。
        has_taq = (ROOT / "TAQ-MOTR-main").exists()
        args.tracker = "taq" if (has_taq and not args.ref_face) else "botsort"
    print(f"参数: tracker={args.tracker} zoom={args.zoom} preset={args.preset} "
          f"max_targets={args.max_targets} min_height={args.min_height} "
          f"height_ratio={args.height_ratio}")
    if str(args.tracker) == "taq":
        print(f"      TAQ: tol={args.taq_tol} 代理高={args.taq_height} "
              f"（1080 上约 9 帧/秒；-tracker botsort 可以先快速看效果）")

    summary = []
    t0 = time.time()
    # args 是共享对象，process_video 会往里写自动匹配的参考图和锁定的 gid。
    # 每个视频开工前还原成"命令行给的值"，否则第一条的参考图会被带到后面所有视频。
    explicit_ref = args.ref_face
    explicit_gid = args.gid
    for v in vids:
        args.ref_face, args.gid = explicit_ref, explicit_gid
        print(f"\n{'=' * 68}")
        try:
            summary.append(process_video(v, args, out_root))
        except SystemExit as exc:
            print(f"[error] {v.name}: {exc}")
            summary.append(dict(slug=slugify(v.stem), video=str(v), error=str(exc)))
        except Exception as exc:                    # noqa: BLE001
            print(f"[error] {v.name}: {type(exc).__name__}: {exc}")
            summary.append(dict(slug=slugify(v.stem), video=str(v),
                                error=f"{type(exc).__name__}: {exc}"))

    write_json(out_root / "batch_summary.json", summary)
    done = [s for s in summary if s.get("status") == "ok"]
    skipped = [s for s in summary if s.get("status") == "skipped"]
    review = [s for s in summary if s.get("status") == "review"]
    failed = [s for s in summary if s.get("error")]
    print(f"\n{'=' * 68}")
    print(f"结果：{len(done)} 完成，{len(skipped)} 跳过，"
          f"{len(review)} 需复核，{len(failed)} 失败")
    print(f"总用时 {(time.time() - t0) / 60:.1f} 分钟")
    for s in summary:
        st = s.get("status")
        if s.get("error"):
            print(f"  ✗ {s['slug']}: {s['error']}")
        elif st == "skipped":
            print(f"  · {s['slug']}: 已跳过（无变化）")
        elif st == "review":
            extra = f"  预览见 outputs/{s['slug']}/_review_preview.jpg" if s.get("preview") else ""
            print(f"  △ {s['slug']}: 需复核（{s.get('reason', '质量过低')}）{extra}")
        elif s.get("targets"):
            warn = "  ⚠ 质量偏低（已出片）" if s.get("quality_notes") else ""
            print(f"  ✓ {s['slug']}: {len(s['targets'])} 个 ID{warn}")
        else:
            print(f"  - {s['slug']}: 无合格目标")

    # 有失败时给非零退出码，bat 能据此显示 FAILED 而不是 DONE。
    if failed:
        raise SystemExit(1)
    if review:
        print("\n提示：有需要复核的结果（上面标 △ 的），"
              "看一下是不是参考脸选错了；换好参考图后重跑一次即可。")

    # 完成后自动打开结果目录，直接拿 MP4/CSV/JPG。
    if done and not args.no_open and out_root.exists():
        try:
            os.startfile(str(out_root))          # noqa: S606
        except OSError:
            print(f"\n结果目录：{out_root}")


if __name__ == "__main__":
    main()
