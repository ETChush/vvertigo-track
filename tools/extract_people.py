# -*- coding: utf-8 -*-
"""从视频里抽几帧，把每一帧的所有人物裁出来，并按人脸去重。

用途：手上只有视频、没有人物照片时，先用它拿到"这个人长什么样"。
挑一张满意的存成 `3.jpg`（和视频同名数字前缀），扔进视频目录，
之后 `run.py` 就能自动匹配成参考图。

    py -3.12 tools/extract_people.py --video 视频.mp4

产物：

    <out>/
      frame_00_00m12s/p01.jpg ... p09.jpg   每帧每个人的裁切（带边距）
      faces/person01.jpg ...                同一个人只留最清晰的一张
      people.txt                            每帧多少人、谁和谁是同一个人
      preview.png                           所有去重人脸的拼图（无损大图），一眼看完

为什么按人脸去重
---------------
一帧里 10 个人，抽 8 帧就是 80 张，人眼根本挑不动。人脸特征的同一人相似度
0.7+、不同人 <0.58，按 0.55 做贪心聚类，80 张能收到十来张。
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.fastvideo import probe_size                      # noqa: E402


def imread_unicode(p: Path):
    """cv2.imread 在 Windows 上读不了中文路径。"""
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except OSError:
        return None


def imwrite_unicode(p: Path, img: np.ndarray, quality: int = 92, fmt: str = ".jpg") -> None:
    """写图。fmt 给 .jpg 用有损 quality，给 .png 走无损（PNG 压缩）。"""
    if fmt == ".png":
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        p.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(p))


def slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", s).strip("_")[:60] or "video"


def pick_times(duration: float, n: int, randomize: bool) -> list[float]:
    """把时长切成 n 段，每段取一点。有覆盖、又不至于全撞在同一秒。

    首尾各留 5%：开头常是黑场/板卡，结尾常是淡出。
    """
    lo, hi = duration * 0.05, duration * 0.95
    if hi <= lo:
        lo, hi = 0.0, max(duration - 0.05, 0.05)
    step = (hi - lo) / max(1, n)
    out = []
    for i in range(n):
        a = lo + i * step
        b = a + step
        t = random.uniform(a, b) if randomize else (a + b) / 2
        out.append(min(max(t, 0.0), max(duration - 0.05, 0.0)))
    return out


def grab(video: Path, t: float, tmp: Path) -> np.ndarray | None:
    p = tmp / f"f{t:.2f}.jpg"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}",
                        "-i", str(video), "-frames:v", "1", "-q:v", "2", str(p)],
                       capture_output=True)
    if r.returncode != 0 or not p.exists():
        return None
    return imread_unicode(p)


def dedup_boxes(boxes, tol: float = 0.35):
    """同一帧里同一个人被重复框住时只留最大的那个。

    密集人群里会出现两个几乎重合的框（实测两张裁切图的人脸余弦 0.992），
    不处理就会把一个人算成两个，聚类直接碎掉。用中心距 vs 框高判断，
    比 IoU 更稳——包含关系（大框套小框）的 IoU 可能只有 0.5。
    """
    boxes = sorted(boxes, key=lambda b: -(b[3] - b[1]))
    keep = []
    for b in boxes:
        bcy = (b[1] + b[3]) / 2
        bcx = (b[0] + b[2]) / 2
        dup = False
        for k in keep:
            kh = max(1.0, k[3] - k[1])
            kcy, kcx = (k[1] + k[3]) / 2, (k[0] + k[2]) / 2
            if abs(bcx - kcx) < tol * kh and abs(bcy - kcy) < tol * kh:
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep


def expand(box, w: int, h: int, pad: float = 0.10):
    """人物框外扩一点，别把手臂/下摆切掉。"""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad * 0.6
    y2 += bh * pad * 0.6
    return (int(max(0, x1)), int(max(0, y1)),
            int(min(w, x2)), int(min(h, y2)))


def contact_sheet(imgs: list[np.ndarray], labels: list[str], cols: int = 6,
                  cell: int = 320) -> np.ndarray | None:
    """把候选拼成一张大图。cell 是每格边长（像素），越大越清晰。

    选人预览要"无损、够大"，所以这里 cell 默认 320（比原来的 160 大 4 倍面积），
    cols 6 列，24 张候选正好 4 行。PNG 无损由调用方用 imwrite_unicode(fmt=".png")。
    """
    if not imgs:
        return None
    rows = (len(imgs) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 24, np.uint8)
    for i, (im, lb) in enumerate(zip(imgs, labels)):
        r, c = divmod(i, cols)
        h, w = im.shape[:2]
        sc = min((cell - 28) / w, (cell - 28) / h)
        th, tw = max(1, int(h * sc)), max(1, int(w * sc))
        thumb = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        y0 = r * cell + 24 + (cell - 28 - th) // 2
        x0 = c * cell + (cell - tw) // 2
        sheet[y0:y0 + th, x0:x0 + tw] = thumb
        cv2.putText(sheet, lb, (c * cell + 6, r * cell + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description="抽样视频并裁出所有人物（按人脸去重）")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="", help="输出目录，默认 <视频名>_people")
    ap.add_argument("--frames", type=int, default=16, help="抽几帧（越多候选越全）")
    ap.add_argument("--max-per-frame", type=int, default=12, help="每帧最多取几个最大的框")
    ap.add_argument("--min-height", type=float, default=0.12,
                    help="框高占画面高的下限，过滤观众/远景")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--weights", default="", help="默认 work/models/yolo11m.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--dedup-thr", type=float, default=0.45,
                    help="人脸聚类阈值；同帧重复是 0.99，跨帧同人一般 0.45-0.7，"
                         "不同人 <0.4。调低会合并更多，但可能把两个人并成一个")
    ap.add_argument("--random", action="store_true", help="每段内随机取点（默认取中点）")
    ap.add_argument("--no-dedup", action="store_true", help="不做人脸聚类，只按帧输出")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"视频不存在：{video}")
    root = Path(__file__).resolve().parents[1]
    weights = args.weights or str(root / "work" / "models" / "yolo11m.pt")
    if not Path(weights).exists():
        raise SystemExit(f"找不到检测权重：{weights}")

    out = Path(args.out) if args.out else video.with_name(f"{slug(video.stem)}_people")
    out.mkdir(parents=True, exist_ok=True)

    w, h, fps = probe_size(str(video))
    duration = 0.0
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(video)], capture_output=True)
    try:
        duration = float(r.stdout.decode().strip().splitlines()[0])
    except (ValueError, IndexError):
        raise SystemExit("读不出视频时长")
    print(f"视频 {video.name}  {w}x{h} @{fps:.2f}  {duration:.1f}s")
    print(f"抽 {args.frames} 帧 → {out}")

    import warnings
    warnings.filterwarnings("ignore")
    from ultralytics import YOLO

    model = YOLO(weights)
    enc = None
    if not args.no_dedup:
        from identity.face import FaceEncoder

        enc = FaceEncoder()

    times = pick_times(duration, args.frames, args.random)
    people: list[dict] = []
    tmp = Path(tempfile.mkdtemp(prefix="vvg_people_"))
    try:
        for i, t in enumerate(times):
            frame = grab(video, t, tmp)
            if frame is None:
                print(f"  [{i + 1}/{len(times)}] {t:6.1f}s  取帧失败")
                continue
            res = model.predict(frame, imgsz=1280, conf=args.conf, classes=[0],
                                device=args.device, verbose=False)[0]
            boxes = [] if res.boxes is None else res.boxes.xyxy.cpu().numpy()
            keep = []
            for b in boxes:
                if (b[3] - b[1]) / max(1, h) < args.min_height:
                    continue
                keep.append(tuple(float(v) for v in b))
            keep = dedup_boxes(keep)
            keep.sort(key=lambda b: -(b[3] - b[1]))
            keep = keep[: args.max_per_frame]

            tag = f"frame_{i:02d}_{int(t // 60):02d}m{int(t % 60):02d}s"
            d = out / tag
            for j, b in enumerate(keep, 1):
                x1, y1, x2, y2 = expand(b, w, h)
                if x2 - x1 < 12 or y2 - y1 < 24:
                    continue
                crop = frame[y1:y2, x1:x2]
                imwrite_unicode(d / f"p{j:02d}.jpg", crop)
                rec = dict(frame=i, t=t, box=b, path=d / f"p{j:02d}.jpg",
                           h=b[3] - b[1], feat=None, face_h=0.0, score=0.0)
                if enc is not None:
                    obs = enc.embed_head(frame, b, up=2)
                    if obs:
                        o = max(obs, key=lambda x: x.score)
                        rec["feat"] = o.feat
                        rec["face_h"] = float(o.h)
                        rec["score"] = float(o.score)
                people.append(rec)
            print(f"  [{i + 1}/{len(times)}] {t:6.1f}s  人物 {len(keep)} 个"
                  f"（有脸 {sum(1 for p in people if p['frame'] == i and p['feat'] is not None)}）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- 按人脸聚类
    lines = [f"video: {video}", f"frames: {len(times)}  people: {len(people)}", ""]
    if args.no_dedup or not any(p["feat"] is not None for p in people):
        lines.append("未做人脸聚类（--no-dedup 或没检出可用人脸）")
        for p in people:
            lines.append(f"  {p['path'].relative_to(out)}")
        (out / "people.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n未做人脸聚类，原始裁切已保存。")
        print(f"产物：{out}")
        return

    clusters: list[dict] = []
    for p in people:
        if p["feat"] is None:
            continue
        best, best_sim = None, -1.0
        for c in clusters:
            s = float(p["feat"] @ c["centroid"])
            if s > best_sim:
                best, best_sim = c, s
        if best is not None and best_sim >= args.dedup_thr:
            best["items"].append(p)
            n = len(best["items"])
            best["centroid"] = (best["centroid"] * (n - 1) + p["feat"]) / n
            best["centroid"] /= np.linalg.norm(best["centroid"]) + 1e-9
        else:
            clusters.append(dict(centroid=p["feat"].copy(), items=[p]))

    # 每人挑一张最清晰的：脸大 + 检测分高
    # 先按出现次数（越多越像主角），再按清晰度
    clusters.sort(key=lambda c: (-len(c["items"]),
                                 -max(p["face_h"] * p["score"] for p in c["items"])))
    faces_dir = out / "faces"
    thumbs, labels = [], []
    lines.append(f"clusters: {len(clusters)}")
    lines.append("（按出现次数排序；人脸的跨帧一致性有限，可能有同一人被拆成多个）")
    lines.append("")
    for k, c in enumerate(clusters, 1):
        best = max(c["items"], key=lambda p: p["face_h"] * p["score"])
        dst = faces_dir / f"person{k:02d}.jpg"
        imwrite_unicode(dst, imread_unicode(best["path"]))
        if k <= 24:
            thumbs.append(imread_unicode(best["path"]))
            labels.append(f"p{k:02d} x{len(c['items'])}")
        frames = sorted({p["frame"] for p in c["items"]})
        lines.append(f"person{k:02d}: {len(c['items'])} 次出现, 帧 {frames}, "
                     f"脸高 {best['face_h']:.0f}px, 来源 {best['path'].name}")
    (out / "people.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sheet = contact_sheet(thumbs, labels)
    if sheet is not None:
        # 无损 PNG + 大格：选人时要看清楚脸，JPEG 压缩 + 160px 格会糊
        imwrite_unicode(out / "preview.png", sheet, fmt=".png")

    print(f"\n去重后 {len(clusters)} 个人")
    for k, c in enumerate(clusters, 1):
        best = max(c["items"], key=lambda p: p["face_h"] * p["score"])
        print(f"  person{k:02d}  {len(c['items']):3d} 次  脸 {best['face_h']:5.0f}px")
    print(f"\n产物：{out}")
    print(f"  预览  {out / 'preview.png'}")
    print(f"  人脸  {faces_dir}\\person01.jpg ...")
    print(f"  明细  {out / 'people.txt'}")
    print("\n挑一张满意的，按视频名改成 3.jpg 放进视频目录，run.py 会自动认成参考图。")


if __name__ == "__main__":
    main()
