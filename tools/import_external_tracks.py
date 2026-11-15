# -*- coding: utf-8 -*-
"""把外部跟踪结果导入成标准 tracks.jsonl，走同一条摄影机/渲染链路。

用途：验证「跟踪器可替换」不是口号。TAQ-MOTR 的输出（MOT txt）导进来后，
identity / camera / render 三层完全不用改。

  python tools/import_external_tracks.py --taq <txt> --outdir outputs/taq
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import VideoInfo, write_json, write_jsonl   # noqa: E402
from core.video import probe                                  # noqa: E402
from scene.shot_detector import detect as detect_shots        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taq", required=True, help="MOT 格式 txt（帧,ID,x,y,w,h,分数）")
    ap.add_argument("--video", default="seg_002.mp4")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--shot-id", type=int, default=0, help="外部结果不分镜头时统一填入")
    ap.add_argument("--single-shot", action="store_true",
                    help="不跑镜头检测（4K 长片上解码整片极易爆内存），全片当一个镜头")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    info = probe(args.video)
    write_json(out / "video_info.json", dict(
        path=info.path, width=info.width, height=info.height, fps=info.fps,
        n_frames=info.n_frames, duration=info.duration,
        has_audio=info.has_audio, codec=info.codec))
    if args.single_shot:
        class _One:
            shot_id, start_frame, start_t, confidence = 0, 0, 0.0, 1.0
            end_frame = int(info.n_frames)
            end_t = float(info.duration)

            def contains(self, fi: int) -> bool:      # noqa: ARG002
                return True

        shots = [_One()]
    else:
        shots = detect_shots(args.video, info)
    write_json(out / "shots.json", [dict(shot_id=s.shot_id, start_frame=s.start_frame,
                                         end_frame=s.end_frame, start_t=s.start_t,
                                         end_t=s.end_t, confidence=s.confidence)
                                    for s in shots])

    def shot_of(fi: int) -> int:
        for s in shots:
            if s.contains(fi):
                return s.shot_id
        return 0

    rows = []
    for line in Path(args.taq).read_text(encoding="utf-8").splitlines():
        q = line.strip().split(",")
        if len(q) < 7:
            continue
        fi = int(float(q[0])) - 1          # MOT 从 1 开始
        lid = int(float(q[1]))
        x, y, w, h = (float(v) for v in q[2:6])
        rows.append(dict(fi=fi, t=round(fi / info.fps, 4), lid=lid,
                         sid=args.shot_id if args.shot_id >= 0 else shot_of(fi),
                         x1=round(x, 1), y1=round(y, 1),
                         x2=round(x + w, 1), y2=round(y + h, 1),
                         s=round(float(q[6]), 3)))
    rows.sort(key=lambda r: (r["fi"], r["lid"]))
    n = write_jsonl(out / "tracks.jsonl", rows)
    write_json(out / "analyze_stats.json", dict(frames=info.n_frames, observations=n,
                                                avg_per_frame=round(n / info.n_frames, 2),
                                                source="external:taq", shots=len(shots)))
    print(f"导入 {n} 个轨迹点 → {out/'tracks.jsonl'}  ({len(shots)} 个镜头)")


if __name__ == "__main__":
    main()
