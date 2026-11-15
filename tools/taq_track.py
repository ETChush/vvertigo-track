# -*- coding: utf-8 -*-
"""用 TAQ-MOTR 跑跟踪，输出 MOT txt（可直接喂给 import_external_tracks）。

为什么单独一个工具，而不是塞进检测器：
TAQ 只认 DanceTrack 目录格式（img1/*.jpg），不认 mp4。而且它**不需要原始分辨率**——
统一抽到高 1080 再跑，检测框按比例还原回源坐标即可。身份（人脸/人体）和渲染本来
就重新读原片，所以这一步不吃画质，只吃磁盘和时间。

这和"整条管线压到 1080"是两件事：压的只是跟踪器输入，点位精度由还原后的源坐标
决定，渲染还是原片。

  python tools/taq_track.py --video 待处理/seg_002.mp4 --seq seg_002
  python tools/taq_track.py --video x.mp4 --seq x --tol 30 --force
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAQ = ROOT / "TAQ-MOTR-main"
sys.path.insert(0, str(ROOT))

from core.video import probe          # noqa: E402


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fa5]+", "_", name)
    return s.strip("_")[:32]


def extract_frames(video: Path, frames_dir: Path, height: int, force: bool) -> int:
    """抽帧到帧高 = height（宽度按比例，保持偶数）。已抽过且数量对得上就复用。"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    have = len(list(frames_dir.glob("*.jpg")))
    info = probe(video)
    if have and not force and have >= info.n_frames * 0.99:
        print(f"[taq] 复用已有帧 {have} 张（{frames_dir}）")
        return have

    for p in frames_dir.glob("*.jpg"):
        p.unlink()
    print(f"[taq] 抽帧 {video.name} → {height}p ...")
    t = time.time()
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video),
           "-vf", f"scale=-2:{height}", "-q:v", "2", "-start_number", "1",
           str(frames_dir / "%08d.jpg")]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
    if r.returncode != 0:
        raise SystemExit(f"抽帧失败：{r.stderr[:500]}")
    n = len(list(frames_dir.glob("*.jpg")))
    print(f"[taq] 抽帧完成 {n} 张，{time.time() - t:.0f}s")
    if abs(n - info.n_frames) > max(2, info.n_frames * 0.01):
        print(f"[warn] 帧数不一致：源 {info.n_frames}，抽出 {n}")
    return n


def make_config(seq: str, root: Path, tol: int, template: str) -> tuple[Path, str]:
    """从模板配置派生一份，只改实验名、视频列表、前缀。"""
    src = TAQ / "configs" / template
    if not src.exists():
        raise SystemExit(f"找不到配置模板：{src}")
    exp = f"{seq}_tol{tol}"
    text = src.read_text(encoding="utf-8")
    text = re.sub(r"^EXP_NAME:.*$", f"EXP_NAME: {exp}", text, count=1, flags=re.M)
    text = re.sub(r"^  eval_videos:.*$", f"  eval_videos: [{seq}]", text, count=1, flags=re.M)
    text = re.sub(r"^  video_name_prefix:.*$", '  video_name_prefix: ""', text,
                  count=1, flags=re.M)
    text = re.sub(r"^  dataset_root:.*$", f"  dataset_root: {root.as_posix()}", text,
                  count=1, flags=re.M)
    cfg_dir = ROOT / "work" / "taq_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg_dir / f"{exp}.yaml"
    cfg.write_text(text, encoding="utf-8")
    return cfg, exp


def _child_env() -> dict:
    """跑 main.py 的环境。把 git 目录补进 PATH。

    TAQ 启动时调 `git rev-parse HEAD` 记版本号，PATH 里没有 git 就直接抛
    FileNotFoundError 把推理带崩（utils.py 那边已改成容错，这里再补一层，
    免得它别的地方还要用 git）。
    """
    import os
    import shutil

    env = dict(os.environ)
    if not shutil.which("git"):
        for c in ("C:/Program Files/Git/cmd", "C:/Program Files (x86)/Git/cmd",
                  "C:/Program Files/Git/bin"):
            if Path(c, "git.exe").exists():
                env["PATH"] = c + os.pathsep + env.get("PATH", "")
                break
    return env


def _result_snapshot() -> set:
    """当前所有结果 txt。跑完取差集，就不用猜 TAQ 把目录名改成什么了。"""
    out = set()
    for pat in ("*/result/*.txt", "*/result/**/*.txt"):
        out |= {p.resolve() for p in (TAQ / "test" / "inference_test").glob(pat)}
    return out


def run_taq(cfg: Path, exp: str, root: Path, ckpt: Path, gpus: str,
            seq: str = "") -> Path:
    cmd = [sys.executable, "main.py", "-E", exp, "-C", str(cfg), "-M", "test",
           "-DR", str(root), "-EFP", str(ckpt.resolve()), "--gpus", gpus]
    print(f"[taq] 推理中（{exp}）...")
    t = time.time()
    before = _result_snapshot()
    r = subprocess.run(cmd, cwd=str(TAQ), text=True, errors="ignore",
                       env=_child_env())
    if r.returncode != 0:
        raise SystemExit(f"TAQ 推理失败（{r.returncode}），看上面的输出")
    print(f"[taq] 推理完成 {time.time() - t:.0f}s")

    # TAQ 会把实验目录名里的非 ASCII 清掉（中文 seq → "sistar_so_co_top_0_1"），
    # 所以不能拿原 seq 去拼路径。优先用"新增了哪些 txt"来判断，最可靠。
    new = sorted(_result_snapshot() - before, key=lambda p: p.stat().st_mtime)
    if new:
        return new[-1]
    return find_result(exp, seq)


def find_result(exp: str, seq: str = "") -> Path:
    """结果写在 test/inference_test/<exp>_<module>_<ckpt>/result/ 下，找最新的那个。

    exp 里可能含中文，而 TAQ 会把**目录名**里的中文删掉、只保留文件名。
    所以按名字找不到时再按 `<seq>.txt` 这个文件名找一遍。
    """
    root = TAQ / "test" / "inference_test"
    cands = sorted(root.glob(f"{exp}*/result/*.txt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands and seq:
        cands = sorted(root.glob(f"*/result/{seq}.txt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        cands = sorted((TAQ / "outputs").glob(f"**/{exp}*/result/*.txt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise SystemExit(f"找不到 {exp} 的结果 txt")
    return cands[0]


def scale_mot(src_txt: Path, out_txt: Path, k: float) -> int:
    """把 1080 代理坐标还原到源分辨率，写回 out_txt。"""
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_txt.open("w", encoding="utf-8") as f:
        for line in src_txt.read_text(encoding="utf-8").splitlines():
            q = line.strip().split(",")
            if len(q) < 7:
                continue
            fi, tid = int(float(q[0])), int(float(q[1]))
            x, y, w, h = (float(v) * k for v in q[2:6])
            f.write(f"{fi},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{q[6]},-1,-1,-1\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="用 TAQ-MOTR 生成 MOT 轨迹")
    ap.add_argument("--video", required=True)
    ap.add_argument("--seq", default="", help="序列名，默认取视频名")
    ap.add_argument("--out", default="", help="输出 MOT txt，默认 work/taq/<seq>.txt")
    ap.add_argument("--root", default="work/TAQdata", help="DanceTrack 数据根目录")
    ap.add_argument("--height", type=int, default=1080, help="抽帧高度（代理分辨率）")
    ap.add_argument("--tol", type=int, default=600, choices=[30, 600],
                    help="丢失容忍帧数：600 = 放宽 10 秒（v3 用的就是这档）")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--force", action="store_true", help="重新抽帧并重跑推理")
    ap.add_argument("--force-frames", action="store_true", help="只重新抽帧")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        video = (Path.cwd() / video) if (Path.cwd() / video).exists() else ROOT / video
    if not video.exists():
        raise SystemExit(f"视频不存在：{video}")

    seq = args.seq or slugify(video.stem)
    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    out = Path(args.out) if args.out else ROOT / "work" / "taq" / f"{seq}.txt"
    if not out.is_absolute():
        out = ROOT / out

    t0 = time.time()
    info = probe(video)
    frames_dir = root / "DanceTrack" / "test" / seq / "img1"
    n = extract_frames(video, frames_dir, args.height, args.force or args.force_frames)

    # 代理帧高与源高之比，用来把框还原回源坐标
    proxy_h = min(int(args.height), info.height)
    k = float(info.height) / float(proxy_h)
    print(f"[taq] 源 {info.width}x{info.height} @{info.fps:g}fps {info.n_frames} 帧"
          f" → 代理高 {proxy_h}（还原系数 {k:.3f}）")

    template = "segtrack_eval_tol600.yaml" if args.tol == 600 else "segtrack_eval.yaml"
    cfg, exp = make_config(seq, root, args.tol, template)
    ckpt = TAQ / "TAQ_MOTR_Dancetrack_ckpt.pth"
    if not ckpt.exists():
        ckpt = TAQ / "TAQ_MOTR_Dancetrack.pth"

    # 同参数的结果比源视频新 → 直接复用。6383 帧要跑 12 分钟，重跑纯属浪费。
    raw = None
    if not args.force:
        try:
            prev = find_result(exp, seq)
            if prev.stat().st_mtime > max(video.stat().st_mtime, frames_dir.stat().st_mtime):
                print(f"[taq] 复用已有结果 {prev.parent.parent.name}/result/{prev.name}")
                raw = prev
        except SystemExit:
            raw = None
    if raw is None:
        raw = run_taq(cfg, exp, root, ckpt, args.gpus, seq)

    nn = scale_mot(raw, out, k)
    print(f"[taq] 轨迹点 {nn} → {out}")
    print(f"[taq] 总用时 {time.time() - t0:.0f}s（{n} 帧）")


if __name__ == "__main__":
    main()
