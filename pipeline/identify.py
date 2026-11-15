# -*- coding: utf-8 -*-
"""Pass 3：身份关联。局部轨迹 → 全局身份 → 目标逐帧序列。

产物：
  identity_tracks.json   每个全局身份的摘要（帧数、覆盖、局部轨迹数）
  identity_events.json   绑定/恢复/新建/可疑换人 的全部证据
  target_tracks.jsonl    目标人物逐帧 bbox 与置信度
  identity_stats.json    标定结果与整体统计
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from core.schema import VideoInfo, read_json, read_jsonl, write_json, write_jsonl
from identity import manager
from identity.appearance import make_embedder


def run(cfg: dict) -> Path:
    out_dir = Path(cfg["output"]["dir"]).resolve()
    info = read_json(out_dir / "video_info.json")
    icfg = cfg["identity"]
    fps = float(info["fps"])
    n_frames = int(info["n_frames"])
    if cfg["input"].get("limit"):
        n_frames = min(n_frames, int(cfg["input"]["limit"]))

    t0 = time.time()
    rows = list(read_jsonl(out_dir / "tracks.jsonl"))
    tracks_all = manager.build_local_tracks(rows)
    # 阈值按源高归一化：<=1 视为比例（0.083 对应 1080p 的 90px），>1 视为像素。
    # 4K/8K 素材上写死像素值会直接失效（4K 的观众就有 180px 高）。
    min_h = float(icfg["min_track_height"])
    if min_h <= 1.0:
        min_h = min_h * float(info["height"])
    tracks = manager.filter_tracks(tracks_all,
                                   int(icfg["min_track_frames"]),
                                   min_h)
    print(f"[pass3] 局部轨迹 {len(tracks_all)} → 保留 {len(tracks)} "
          f"(≥{icfg['min_track_frames']} 帧, 高≥{min_h:.0f}px = 源高 {100*min_h/float(info['height']):.1f}%)")

    embedder = make_embedder(icfg.get("embedder", "resnet18"))
    vinfo = VideoInfo(path=info["path"], width=info["width"], height=info["height"],
                      fps=fps, n_frames=n_frames, duration=info["duration"],
                      has_audio=info["has_audio"], codec=info["codec"])

    fcfg = (icfg.get("face") or {})
    pcfg = (icfg.get("person_analysis") or {})
    face_enc = None
    analyzer = None
    if fcfg.get("enabled", True):
        try:
            from identity.face import FaceEncoder
            face_enc = FaceEncoder()
        except Exception as exc:      # noqa: BLE001
            print(f"[warn] 人脸编码器不可用（{exc}）")
    if pcfg.get("enabled", True):
        try:
            from identity.person_analysis import PersonAnalyzer
            analyzer = PersonAnalyzer(name=pcfg.get("name", "cheetah_s"))
        except Exception as exc:      # noqa: BLE001
            print(f"[warn] 人体 ReID 不可用（{type(exc).__name__}: {exc}）")

    # 三路特征一次解码完成（4K/8K 上解码是瓶颈，分开跑等于三倍成本）
    # 高分辨率素材按比例放宽采样间隔：采样密度要按"时间"一致，不是按帧数。
    # 4K 是 1080p 的 2 倍高，同样的 every=4 会付 2 倍的检测成本却不增加信息。
    res_scale = max(1.0, float(info["height"]) / 1080.0)
    emb_every = int(icfg.get("sample_every", 4))
    face_every = max(1, int(round(int(fcfg.get("sample_every", 4)) * res_scale)))
    pa_every = max(1, int(round(int(pcfg.get("every", 30)) * res_scale)))
    st = manager.attach_all(vinfo.path, vinfo, tracks, embedder=embedder,
                            face_encoder=face_enc, analyzer=analyzer,
                            emb_every=emb_every, face_every=face_every,
                            pa_every=pa_every,
                            max_samples=int(icfg.get("max_samples", 12)))
    print(f"[pass3] 采样间隔（源高 {info['height']}） 外观每 {emb_every} 帧 / "
          f"人脸每 {face_every} / 人体每 {pa_every}")
    print(f"[pass3] 特征完成 {time.time()-t0:.1f}s  编码器 {embedder.name}  "
          f"外观 {st['tracks_with_emb']}/{len(tracks)}  人脸 {st['tracks_with_face']}  "
          f"人体 {st['tracks_with_body']}")

    face_cal = manager.calibrate_face(tracks) if st["tracks_with_face"] else \
        dict(pos=0.62, neg=0.55, calibrated=False, n_between=0)
    if st["tracks_with_face"]:
        print(f"[pass3] 人脸阈值 pos={face_cal['pos']} neg={face_cal['neg']}  "
              f"(共现 p95={face_cal.get('between_p95')})")

    body_cal = manager.calibrate_body2(tracks) if st["tracks_with_body"] else \
        dict(pos=0.72, neg=0.62, calibrated=False)
    if st["tracks_with_body"]:
        print(f"[pass3] 人体阈值 pos={body_cal['pos']} neg={body_cal['neg']}  "
              f"(共现 p50={body_cal.get('p50')} p95={body_cal.get('p95')})")
    if analyzer is not None:
        analyzer.close()

    # ---- 按外观突变切段：同一条 BoT 轨迹中途换人时，必须切成两段分别认人
    from identity import refine
    tracks, split_stats = refine.split(tracks)
    print(f"[pass3] 轨迹切段 {split_stats['input']} → {split_stats['output']} "
          f"(切开 {split_stats['split']} 条)")

    low_cal = manager.calibrate_low(tracks)
    print(f"[pass3] 下半身颜色阈值 neg={low_cal['neg']}  "
          f"(共现 p50={low_cal.get('p50')} p99={low_cal.get('p99')})")

    cal = manager.calibrate(tracks)
    thr = float(icfg.get("appearance_thresh") or 0) or cal["thresh"]
    print(f"[pass3] 阈值标定 {thr:.3f}  "
          f"(同轨迹 p10={cal.get('within_p10')} / 共现 p99={cal.get('between_p99')} "
          f"| n_within={cal['n_within']} n_between={cal['n_between']})")

    ident, events = manager.cluster(tracks, thr,
                                    motion_gate=float(icfg.get("motion_gate", 2.2)),
                                    min_new_frames=int(icfg.get("new_identity_min_frames", 30)),
                                    fps=fps,
                                    face_pos=float(fcfg.get("pos") or face_cal["pos"]),
                                    face_neg=float(fcfg.get("neg") or face_cal["neg"]),
                                    low_neg=float(low_cal["neg"]),
                                    body_pos=float(body_cal["pos"]),
                                    body_neg=float(body_cal["neg"]))
    swaps = manager.detect_swaps(tracks, thr, fps=fps)
    events.extend(swaps)

    # 切段后的轨迹先落盘，再做选人。批量处理时"选人失败"只应该影响这条视频的
    # 目标产物，不应该让前面几分钟的分析结果一起丢掉（run.py 会接着自己选人）。
    write_jsonl(out_dir / "tracks_kept.jsonl", (
        dict(sid=k[0], lid=k[1], gid=t.global_id, fi=fi,
             x1=round(b[0], 1), y1=round(b[1], 1), x2=round(b[2], 1), y2=round(b[3], 1),
             s=round(sc, 3))
        for k, t in tracks.items()
        for fi, b, sc in zip(t.frames, t.boxes, t.scores)))

    tcfg = cfg["target"]
    ref_face = None
    if tcfg.get("face_image"):
        from identity.face import FaceEncoder
        ref_face = FaceEncoder().embed_file(tcfg["face_image"])
        print(f"[pass3] 参考人脸: {tcfg['face_image']}")
    gid = manager.select_target(ident, tracks, mode=tcfg.get("auto_pick", "largest"),
                                box=tcfg.get("box") or None,
                                point=tcfg.get("point") or None,
                                at_frame=int(tcfg.get("at_frame", -1)),
                                gid=int(tcfg.get("global_id", 0) or 0),
                                ref_face=ref_face)
    if gid < 0:
        raise SystemExit("没能选出目标人物")
    series, summary = manager.build_target_series(ident, tracks, gid, n_frames, fps)

    # 目标局部轨迹里出现过的所有 ID，用于报告"跨遮挡是否保住身份"
    tgt_tracks = [k for k in tracks if tracks[k].global_id == gid]
    summary["local_track_ids"] = [f"shot{k[0]}/id{k[1]}" for k in tgt_tracks]

    write_json(out_dir / "identity_tracks.json",
               [dict(gid=g.gid, frames=g.frames, tracks=len(g.track_keys),
                     first_frame=min(g.frames_all), last_frame=max(g.frames_all),
                     first_t=round(min(g.frames_all) / fps, 2),
                     last_t=round(max(g.frames_all) / fps, 2))
                for g in sorted(ident, key=lambda x: -x.frames)])
    write_json(out_dir / "identity_events.json", [e.to_dict() for e in events])

    write_json(out_dir / "local_tracks.json", [
        dict(sid=k[0], lid=k[1], gid=t.global_id, n=t.n,
             start=t.start, end=t.end, h=round(t.median_h, 1),
             cx=round(float(np.median([(b[0] + b[2]) / 2 for b in t.boxes])), 1),
             face_n=t.face_n, face_cons=round(t.face_cons, 3))
        for k, t in tracks.items()])
    write_json(out_dir / "identity_faces.json", {
        str(g.gid): [round(float(v), 5) for v in
                     (np.mean(np.stack(g.face_gallery), axis=0)
                      / (np.linalg.norm(np.mean(np.stack(g.face_gallery), axis=0)) + 1e-9))]
        for g in ident if g.face_gallery})

    write_jsonl(out_dir / "target_tracks.jsonl", (
        dict(fi=s.fi, t=round(s.t, 4), gid=s.global_id, lid=s.local_id,
             x1=round(s.x1, 1), y1=round(s.y1, 1), x2=round(s.x2, 1), y2=round(s.y2, 1),
             conf=round(s.identity_conf, 3), sid=s.shot_id)
        for s in series if s is not None))
    write_json(out_dir / "identity_stats.json", dict(
        threshold=thr, calibration=cal, face=face_cal, lowbody=low_cal,
        person_analysis=st, body_reid=body_cal,
        tracks_with_face=st['tracks_with_face'], n_local_tracks=len(tracks),
        n_identities=len(ident), target=summary,
        events_total=len(events),
        swap_suspects=len(swaps),
        seconds=round(time.time() - t0, 1),
        embedder=embedder.name))

    print(f"[pass3] 全局身份 {len(ident)} 个  目标 = ID {gid}  "
          f"覆盖 {summary['coverage']*100:.1f}%  "
          f"最长空洞 {summary['longest_gap_s']}s  可疑换人 {len(swaps)}")
    for g in sorted(ident, key=lambda x: -x.frames)[:6]:
        tag = "  <== TARGET" if g.gid == gid else ""
        print(f"        ID {g.gid:>2}: {g.frames:5d} 帧  "
              f"{min(g.frames_all)/fps:6.1f}-{max(g.frames_all)/fps:6.1f}s  "
              f"局部轨迹 {len(g.track_keys)}{tag}")
    return out_dir
