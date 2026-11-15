# -*- coding: utf-8 -*-
"""为多个全局身份批量生成 9:16 直拍成片。

每个身份：选目标 -> 规划取景 -> 渲染。串行执行，因为中间产物共用同一个 outdir。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent


def run(args: list[str]) -> tuple[bool, str]:
    p = subprocess.run([PY, "main.py", *args], cwd=ROOT,
                       capture_output=True, text=True, errors="ignore")
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode == 0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/taq600")
    ap.add_argument("--ids", default="1,2,3,4,5,10")
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--preset", default="smooth")
    ap.add_argument("--prefix", default="seg_002_dancer")
    ap.add_argument("--dest", default=".")
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    ok_list, fail = [], []
    for gid in ids:
        t0 = time.time()
        print(f"\n=== ID {gid} ===", flush=True)
        ok, log = run(["--stage", "target", "--outdir", args.outdir,
                       "--set", f"target.global_id={gid}"])
        if not ok:
            print("  target 失败:", log.strip().splitlines()[-3:])
            fail.append(gid)
            continue
        m = [ln for ln in log.splitlines() if "可见" in ln or "轨迹" in ln]
        for ln in m[-2:]:
            print("  " + ln.strip())

        ok, log = run(["--stage", "plan,render", "--outdir", args.outdir,
                       "--set", f"camera.zoom={args.zoom}",
                       "--set", f"camera.preset={args.preset}"])
        if not ok:
            print("  plan/render 失败:", log.strip().splitlines()[-3:])
            fail.append(gid)
            continue
        for ln in log.splitlines():
            if "反向" in ln or "全身" in ln:
                print("  " + ln.strip())

        src = Path(ROOT) / args.outdir / "final.mp4"
        dst = dest / f"{args.prefix}{gid:02d}_zoom{args.zoom:g}_{args.preset}.mp4"
        shutil.copy2(src, dst)
        mb = dst.stat().st_size / 1e6
        print(f"  -> {dst.name}  {mb:.0f}MB  {time.time()-t0:.0f}s")
        ok_list.append((gid, dst.name))

    print(f"\n成功 {len(ok_list)} 条，失败 {len(fail)} 条 {fail if fail else ''}")
    for gid, name in ok_list:
        print(f"  ID {gid:>2}: {name}")


if __name__ == "__main__":
    main()
