# -*- coding: utf-8 -*-
"""把整个文件夹的视频批量做成"每人一条竖屏直拍"。

对每个视频：
  1. analyze   检测 + 跟踪（YOLO + BoT-SORT）
  2. identify  身份聚类（人脸 + 人体 ReID + 切段）
  3. 按覆盖时长挑出主要人物，逐个规划取景
  4. 一次解码、多路渲染（4K/8K 上解码是瓶颈，不能每人渲一遍）

用法：
  python tools/process_folder.py --src 待处理 --zoom 2.0 --preset smooth
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camera import planner                       # noqa: E402
from core import config                          # noqa: E402
from core.schema import read_json, read_jsonl, write_json   # noqa: E402
from export import render                        # noqa: E402
from identity import manager                     # noqa: E402
from pipeline import analyze, identify            # noqa: E402


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fa5]+", "_", name)
    return s.strip("_")[:48]


def load_identities(out_dir: Path):
    """从 tracks_kept.jsonl 重建 global id → 局部轨迹。"""
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


def process_video(v: Path, args) -> dict:
    cfg = config.load(str(ROOT / "configs/default.yaml"))
    cfg["input"]["video"] = str(v)
    cfg["input"]["limit"] = args.limit
    slug = slugify(v.stem)
    out_dir = ROOT / "outputs" / slug
    cfg["output"]["dir"] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    has_tracks = (out_dir / "tracks.jsonl").exists()
    has_kept = (out_dir / "tracks_kept.jsonl").exists()
    if has_kept:
        print(f"\n{'='*70}\n[{slug}] 复用已有分析结果")
    else:
        if not has_tracks or args.redo_all:
            print(f"\n{'='*70}\n[{slug}] analyze")
            t = time.time()
            analyze.run(cfg, force=True)
            print(f"[{slug}] analyze {time.time()-t:.0f}s")
        else:
            print(f"\n{'='*70}\n[{slug}] 复用检测结果，只重跑身份")
        print(f"[{slug}] identify")
        t = time.time()
        identify.run(cfg)
        print(f"[{slug}] identify {time.time()-t:.0f}s")

    info = read_json(out_dir / "video_info.json")
    fps = float(info["fps"])
    n_frames = int(info["n_frames"])
    if args.limit:
        n_frames = min(n_frames, args.limit)
    tracks, ident, by_gid = load_identities(out_dir)

    # 挑主要人物：覆盖够长 + 框高够大（按源高比例，兼容 4K/8K）
    min_h = float(cfg["identity"]["min_track_height"])
    if min_h <= 1.0:
        min_h = min_h * float(info["height"])
    cands = []
    for gid in by_gid:
        ts = by_gid[gid]
        h = float(np.median([t.median_h for t in ts]))
        cov = len({fi for t in ts for fi in t.frames})
        if h >= min_h * 1.6 and cov >= args.min_coverage * n_frames:
            cands.append((cov, h, gid))
    cands.sort(reverse=True)
    # 按画面里最高的人分层：观众在远景，框高通常只有台上的一半。
    # 只用绝对阈值在高分辨率素材上分不开（4K 的观众也有 800px）。
    if cands:
        hmax = max(h for _, h, _ in cands)
        floor = args.height_ratio * hmax
        cands = [c for c in cands if c[1] >= floor]
    picked = cands[: args.max_dancers]
    print(f"[{slug}] 候选人物 {len(cands)} 个，渲染前 {len(picked)} 个："
          + ", ".join(f"ID{g}({c/60:.0f}s,h{h:.0f})" for c, h, g in picked))

    ccfg = dict(cfg["camera"])
    ccfg.update(zoom=args.zoom, preset=args.preset, out_width=int(cfg["export"]["width"]))
    body_min = float(cfg["identity"].get("body_min_ref", 0.64))

    jobs, stats = [], []
    for cov, h, gid in picked:
        series, summary = manager.build_target_series(ident, tracks, gid, n_frames, fps,
                                                      body_min=body_min)
        # zoom 按"这个人自己有多大"来定，不是固定 2 倍。
        # 用 p90 框高（跳跃/近景的最坏情况）保证绝大多数帧能装下全身；
        # 再用用户给的 zoom 封顶——想更紧就切脚，那是取舍不是 bug。
        hs = np.array([s.y2 - s.y1 for s in series if s is not None], dtype=float)
        h_p90 = float(np.percentile(hs, 90)) if len(hs) else h
        need_h = h_p90 * (1.0 + 2.0 * float(ccfg.get("vertical", {}).get("headroom", 0.06)))
        zoom_i = float(np.clip(float(info["height"]) / max(need_h, 1.0), 1.0, args.zoom))
        ccfg_i = dict(ccfg, zoom=zoom_i)
        poses, cam = planner.plan(series, info, ccfg_i, gid)
        out = out_dir / f"dancer_gid{gid:02d}.mp4"
        jobs.append(dict(out=str(out), poses=poses))
        stats.append(dict(gid=gid, out=str(out), coverage=summary["coverage"],
                          longest_gap_s=summary["longest_gap_s"],
                          median_h=round(h, 1), h_p90=round(h_p90, 1),
                          zoom=round(zoom_i, 3), crop=f"{cam['crop_w']}x{cam['crop_h']}",
                          upscale=cam["upscale"], full_body=cam["target_inside_ratio"],
                          reversals=cam["reversals"],
                          keyframes=cam["path"]["keyframes"]))
        print(f"  ID{gid:>2}: 覆盖 {summary['coverage']*100:5.1f}%  空洞 "
              f"{summary['longest_gap_s']:5.2f}s  框高p90 {h_p90:4.0f}  "
              f"zoom {zoom_i:.2f} 裁切 {cam['crop_w']}x{cam['crop_h']}  "
              f"放大 {cam['upscale']}x  全身 {cam['target_inside_ratio']*100:.1f}%")

    if jobs:
        print(f"[{slug}] render × {len(jobs)}")
        t = time.time()
        res = render.render_multi(str(v), jobs, fps,
                                  out_w=int(cfg["export"]["width"]),
                                  out_h=int(cfg["export"]["height"]),
                                  codec=cfg["export"].get("codec", "h264_nvenc"),
                                  quality=int(cfg["export"].get("quality", 19)),
                                  audio=bool(cfg["export"].get("preserve_audio", True)))
        print(f"[{slug}] render {time.time()-t:.0f}s")
        for r, s in zip(res, stats):
            s["size_mb"] = r["size_mb"]
        write_json(out_dir / "dancers.json", stats)

    print(f"[{slug}] 完成，用时 {time.time()-t_all:.0f}s")
    return dict(slug=slug, video=str(v), info=dict(width=info["width"], height=info["height"],
                                                   fps=info["fps"], n_frames=info["n_frames"]),
                dancers=stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="待处理")
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--preset", default="smooth")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-dancers", type=int, default=8)
    ap.add_argument("--min-coverage", type=float, default=0.12,
                    help="至少要覆盖全片多少比例才算主要人物")
    ap.add_argument("--height-ratio", type=float, default=0.60,
                    help="框高低于画面最高者的该比例，视为远景观众")
    ap.add_argument("--only", default="", help="只处理文件名包含该串的视频")
    ap.add_argument("--redo-all", action="store_true", help="连检测一起重跑")
    args = ap.parse_args()

    src = (ROOT / args.src) if not Path(args.src).is_absolute() else Path(args.src)
    vids = sorted(p for p in src.glob("*.mp4") if p.is_file())
    if args.only:
        vids = [v for v in vids if args.only in v.name]
    print(f"待处理 {len(vids)} 个视频")
    for v in vids:
        print(f"  - {v.name}")

    all_stats = []
    for v in vids:
        try:
            all_stats.append(process_video(v, args))
        except Exception as exc:      # noqa: BLE001
            print(f"[error] {v.name}: {type(exc).__name__}: {exc}")
    write_json(ROOT / "outputs" / "batch_summary.json", all_stats)
    print("\n全部完成")


if __name__ == "__main__":
    main()
