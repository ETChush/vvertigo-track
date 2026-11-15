# -*- coding: utf-8 -*-
"""选人阶段（可反复重跑，不重算任何模型）。

识别阶段已把局部轨迹、全局 ID、人脸库落盘。这里只做「选谁」这一件事，
所以切换目标、换参考脸、改点选坐标都是秒级。

  python main.py --stage target,plan,render --set target.global_id=5
  python main.py --stage target,plan,render --set target.face_image=work/ref_face.jpg
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.schema import read_json, read_jsonl, write_json, write_jsonl
from identity import manager


def run(cfg: dict) -> Path:
    out_dir = Path(cfg["output"]["dir"]).resolve()
    info = read_json(out_dir / "video_info.json")
    fps = float(info["fps"])
    n_frames = int(info["n_frames"])
    if cfg["input"].get("limit"):
        n_frames = min(n_frames, int(cfg["input"]["limit"]))

    kept_path = out_dir / "tracks_kept.jsonl"
    src = kept_path if kept_path.exists() else (out_dir / "tracks.jsonl")
    tracks = manager.build_local_tracks(read_jsonl(src))
    if kept_path.exists():                    # 已切段并分配过全局 ID
        for r in read_jsonl(kept_path):
            tracks[(int(r["sid"]), int(r["lid"]))].global_id = int(r["gid"])
    else:
        for r in read_json(out_dir / "local_tracks.json"):
            k = (int(r["sid"]), int(r["lid"]))
            if k in tracks:
                tracks[k].global_id = int(r["gid"])

    # 重建全局身份（只保留选人需要的字段）
    by_gid: dict[int, list] = {}
    for k, t in tracks.items():
        if t.global_id > 0:
            by_gid.setdefault(t.global_id, []).append(t)
    faces = read_json(out_dir / "identity_faces.json") if (
        out_dir / "identity_faces.json").exists() else {}
    ident = []
    for gid, ts in by_gid.items():
        f = faces.get(str(gid))
        ident.append(manager.GlobalIdentity(
            gid=gid, gallery=[], track_keys=[t.key for t in ts],
            frames_all=[fi for t in ts for fi in t.frames],
            face_gallery=[np.array(f, dtype=np.float32)] if f else []))
    ident.sort(key=lambda g: -g.frames)

    tcfg = cfg["target"]
    ref_face = None
    if tcfg.get("face_image"):
        from identity.face import FaceEncoder
        ref_face = FaceEncoder().embed_file(tcfg["face_image"])
    gid = manager.select_target(ident, tracks, mode=tcfg.get("auto_pick", "largest"),
                                box=tcfg.get("box") or None,
                                point=tcfg.get("point") or None,
                                at_frame=int(tcfg.get("at_frame", -1)),
                                gid=int(tcfg.get("global_id", 0) or 0),
                                ref_face=ref_face)
    if gid < 0:
        raise SystemExit("没能选出目标人物")

    body_min = float((cfg.get("identity") or {}).get("body_min_ref", 0.64))
    series, summary = manager.build_target_series(ident, tracks, gid, n_frames, fps,
                                                  body_min=body_min)
    summary["local_track_ids"] = [f"shot{k[0]}/id{k[1]}" for k, t in tracks.items()
                                  if t.global_id == gid]
    summary["select"] = dict(mode=tcfg.get("select"), gid=gid,
                             face_image=tcfg.get("face_image") or None)

    write_jsonl(out_dir / "target_tracks.jsonl", (
        dict(fi=s.fi, t=round(s.t, 4), gid=s.global_id, lid=s.local_id,
             x1=round(s.x1, 1), y1=round(s.y1, 1), x2=round(s.x2, 1), y2=round(s.y2, 1),
             conf=round(s.identity_conf, 3), sid=s.shot_id)
        for s in series if s is not None))
    write_json(out_dir / "target_summary.json", summary)

    print(f"[target] ID {gid}  可见 {summary['coverage']*100:.1f}%  "
          f"最长空洞 {summary['longest_gap_s']}s  局部轨迹 {summary['local_tracks']} 条")
    print(f"[target] 轨迹: {', '.join(summary['local_track_ids'])}")
    return out_dir
