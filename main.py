# -*- coding: utf-8 -*-
"""VVERTIGO 命令行入口。

  python main.py --stage all                     # 全流程
  python main.py --stage analyze --limit 600     # 只跑检测+跟踪
  python main.py --stage plan                    # 只重算取景（不重跑模型）
  python main.py --stage render                  # 只重渲染
  python main.py --stage target,plan,render --set target.face_image=work/ref_face.jpg
  python main.py --set camera.max_speed=600 --stage plan,render
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import config as cfgmod          # noqa: E402
from core.schema import read_json         # noqa: E402

STAGES = ("analyze", "identify", "target", "plan", "render", "report")


def main() -> None:
    ap = argparse.ArgumentParser(description="VVERTIGO 自动直拍管线")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stage", default="all",
                    help="all 或逗号分隔: analyze,identify,plan,render,report")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="覆盖配置，可多次，如 --set camera.max_speed=600")
    ap.add_argument("--video", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="强制重跑 analyze")
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    if args.video:
        cfg["input"]["video"] = args.video
    if args.outdir:
        cfg["output"]["dir"] = args.outdir
    if args.limit is not None:
        cfg["input"]["limit"] = args.limit
    for kv in args.set:
        k, _, v = kv.partition("=")
        cfgmod.set_path(cfg, k.strip(), v.strip())

    stages = STAGES if args.stage == "all" else tuple(
        s.strip() for s in args.stage.split(",") if s.strip())
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"未知阶段: {unknown}")

    out_dir = Path(cfg["output"]["dir"]).resolve()
    t0 = time.time()
    timings = {}

    if "analyze" in stages:
        from pipeline import analyze
        t = time.time()
        analyze.run(cfg, force=args.force)
        timings["analyze"] = round(time.time() - t, 1)

    if "identify" in stages:
        from pipeline import identify
        t = time.time()
        identify.run(cfg)
        timings["identify"] = round(time.time() - t, 1)

    if "target" in stages:
        from pipeline import target as target_stage
        t = time.time()
        target_stage.run(cfg)
        timings["target"] = round(time.time() - t, 1)

    if "plan" in stages:
        from pipeline import plan
        t = time.time()
        plan.run(cfg)
        timings["plan"] = round(time.time() - t, 1)

    render_stats = None
    if "render" in stages:
        from export import render as renderer
        from pipeline.plan import load_series
        from core.schema import CameraPose

        info = read_json(out_dir / "video_info.json")
        poses_raw = read_json(out_dir / "camera_path.json")
        poses = [CameraPose(**p) for p in poses_raw]
        ecfg = cfg["export"]
        t = time.time()
        out_file = out_dir / "final.mp4"
        render_stats = renderer.render(
            info["path"], poses, out_file,
            fps=float(info["fps"]),
            out_w=int(ecfg["width"]), out_h=int(ecfg["height"]),
            codec=str(ecfg.get("codec", "h264_nvenc")),
            quality=int(ecfg.get("quality", 19)),
            scale_flags=str(ecfg.get("scale_flags", "lanczos")),
            audio=bool(ecfg.get("preserve_audio", True)),
            limit=int(cfg["input"].get("limit") or 0))
        timings["render"] = round(time.time() - t, 1)
        print(f"[pass5] {render_stats}")

        if "report" in stages:
            from evaluation import quality
            n_frames = len(poses)
            series = load_series(out_dir / "target_tracks.jsonl", n_frames)
            quality.contact_sheet(info["path"], poses, out_dir / "contact_sheet.jpg")
            quality.framing_check(info["path"], poses, series, out_dir / "framing_check.jpg")
            print("[pass5] contact_sheet.jpg / framing_check.jpg")

    if "report" in stages:
        from evaluation import quality
        rep = quality.report(out_dir, render_stats)
        print("\n=== 质量报告 ===")
        for line in rep.get("summary_lines", []):
            print("  " + line)

    timings["total"] = round(time.time() - t0, 1)
    print(f"\n阶段耗时: {timings}")
    print(f"产物目录: {out_dir}")


if __name__ == "__main__":
    main()
