# VVERTIGO Track

输入横屏视频 + 一张人脸图，输出**那个人的 9:16 竖屏直拍**，以及可以导进 AE 打关键帧的**点位 CSV**。

```
视频 ──快速检测──> 每帧人体框 ──人脸锚点──> 她的逐帧位置 ──取景规划──> 竖屏成片 + CSV
```

---

## 快速开始

```powershell
cd <项目目录>
```

**双击 `一键出片.bat`**，或：

```powershell
py -3.12 run.py --src 待处理
```

把视频丢进 `待处理/`。参考图**自动匹配**：视频名 `3-00.01.56-....mp4` 就找同目录的 `3.jpg`。

想手动指定：

```powershell
py -3.12 run.py --src D:\clips --ref-face D:\clips\张三.jpg
```

产物：

```
outputs/<视频名>/
  ID1/
    3-ID1.mp4     1080×1920 竖屏直拍（带原音轨）
    3-ID1.csv     点位，源视频像素坐标，导入 AE
    3-ID1.jpg     该人的人脸图（可以直接当下一轮的参考图）
  manifest.json   覆盖/全身率/关键帧/曲线误差/被丢弃的人
  _work/          中间产物，断点续跑靠它
```

---

## 只有视频、没有人物照片时

先跑这个，把视频里的人挑出来：

```powershell
py -3.12 tools/extract_people.py --video 待处理\3-xxx.mp4
```

均匀抽 10 帧（每段内随机取点，避开首尾），检测每帧的所有人，裁出来，再按人脸去重。

```
3-xxx_people/
  frame_00_00m05s/p01.jpg ... p10.jpg   每帧每个人的裁切（外扩 10% 边距）
  faces/person01.jpg ...                去重后每人一张（最清晰的那张）
  preview.jpg                           拼图预览，一眼看完
  people.txt                            出现次数、来源帧、脸高
```

挑一张满意的，**按视频名改成 `3.jpg`** 放进视频目录，之后 `run.py` 就能自动认成参考图。

几点说明：

- **去重按人脸做，跨帧一致性有限。** 背对、侧脸、被挡的帧聚不到一起，同一个人可能被拆成好几个
  `personNN`。所以结果按**出现次数**排序——次数越多越像主角。候选多不要紧，挑一张就行。
- 同帧里同一个人可能被检出两个几乎重合的框（实测两张裁切的人脸余弦 0.992），脚本会先去重，
  否则一个人算两个，聚类直接碎掉。
- `--dedup-thr 0.35` 合并得更狠，但会开始把不同的人并在一起（实测一簇里塞进 15 个裁切，
  而 6 帧最多只能出现 6 次）。默认 0.45 是保守值。
- `--no-dedup` 完全不聚类，只按帧输出原始裁切，配 `preview.jpg` 自己看。
- 密集群舞时裁切框常把旁边的人也带进来，这是检测框本身的问题，脚本只能少外扩一点。

---

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--src` | `待处理` | 视频目录 |
| `--out` | `outputs` | 输出目录 |
| `--ref-face` | 自动 | 参考人脸图；不给就按视频名数字前缀找 |
| `--max-targets` | 3 | 最多出几个人（给参考图时只需要 1 个） |
| `--face-thr` | 0.55 | 判定"这是她"的相似度阈值 |
| `--face-topk` | 10 | 每帧检查几个最大的框。**群舞场景别低于 10**，画面 11 个人时取 6 会漏掉目标 |
| `--face-every` | 8 | 每几帧采一次人脸锚点 |
| `--face-up` | 2 | 头部裁出后放大倍数；1 快 3 倍但相似度从 0.687 掉到 0.672、覆盖掉 10 个点 |
| `--det-w` | 1280 | 检测输入宽度。1280 约 46 FPS；960 约 82 FPS 但覆盖掉约 9 个点；0 = 源分辨率（4K 只有 4 FPS，别用） |
| `--render-engine` | ffmpeg | ffmpeg = 内部逐帧裁剪（约 120 FPS）；python = Python 里裁（约 32 FPS，像素级精确） |
| `--preset` | smooth | follow / smooth / steady，取景平滑程度 |
| `--zoom` | 1.0 | 1.0 = 9:16 最大裁切，最清晰；>1 收紧构图并放大 |
| `--anchor-gap` | 150 | 两个锚点相隔超过这么多帧就不桥接，留空让相机 HOLD |
| `--anchor-grow` | 60 | 单次框关联最多补多少帧，防止靠惯性漂走 |
| `--eps` / `--tol` | 3.0 / 1.5 | CSV 稀疏化容差（源像素）/ 允许的曲线误差（成片像素） |
| `--redo-all` | 关 | 忽略缓存重跑全流程 |

---

## 点位 CSV 与 AE

坐标一律是**源视频像素**（不是成片像素）——插件可能在 AE 里放原片，用成片坐标在 4K 上会差 3.5 倍。

```
# vvertigo-track v1
# source=3.mp4 width=3840 height=2160 fps=59.94 frames=2583
# crop_w=1214 crop_h=2158 out_w=1080 out_h=1920
# scale=0.889621 coord=center,px,source_top_left
# target=ID1 gid=-1 coverage=0.552 median_h=1281
# keyframes=413 frames_per_keyframe=6.3 max_error_px=1.46
frame,time,x,y,crop_w,crop_h,scale,conf,state,kf,tix,tiy,tox,toy
0,0.000,1920.5,1080.0,1214,2158,0.890,0.98,TRACKING,1,0,0,0,0
```

- `x,y` = 裁切框中心（源像素）
- `scale` = out_w / crop_w，AE 里 `Scale = scale × 100`
- `kf=1` 是稀疏关键帧，`tix/tiy/tox/toy` 是它前后的贝塞尔手柄偏移
- 切线从**最终路径**反推（低通 + 冻结 + 软限幅之后），不是在检测点上打关键帧——否则 HOLD 段的手柄还在"动"，AE 里会慢慢漂
- 导出时会用这些关键帧重算一遍曲线，和原路径的最大偏差写进 `max_error_px`（默认 1.5 成片像素内）

**AE 插件**：`ae/VVERTIGO_Track.jsx`

1. AE 里 `文件 > 脚本 > 运行脚本文件…`
2. 合成设为 1080×1920，把原片拖进去，选中图层
3. 选 CSV，点导入

常用的话丢进 `Support Files/Scripts/ScriptUI Panels/`，从「窗口」菜单打开成常驻面板。
两种模式：稀疏关键帧（可手改、带贝塞尔手柄）/ 逐帧烘焙（像素级精确）。

---

## 算法要点（为什么这么做）

拆成五步，每一步都是被某个具体问题逼出来的：

**1. 检测只要框，不要轨迹。**
人脸锚点这条路不需要跟踪器的 ID 连续性，只要求每帧有准的人体框。所以跳过 BoT-SORT/TAQ，直接 ffmpeg 管道 + 批量 YOLO。4K AV1 上从 4.3 FPS 提到 38–84 FPS。

**2. 人脸锚点，不信任长轨迹。**
多人素材上跟踪器的长轨迹一定会串人。实测某条轨迹在 5.9s 是左边某个人（人脸相似度 0.044），120s 是她（0.709），183s 又是别人。事后拿人脸去核验只能"发现"错，不能变对。所以只相信两件事：脸转过来了的那些帧，以及转身后 1.5 秒内框的连续性。

**3. 接不上就留空。**
锚点之间用检测框做短距离关联（位置 + 尺寸 + **宽高比**连续性）。超过 `--anchor-grow` 就停，留空让相机 HOLD。宁可短暂保持原取景，也不要跟着别人走。

**4. 双门槛选人。**
框高 < 画面高 20% → 判为观众/背景，不建 ID；再过相对门槛（< 最高者的 55%）。过了门槛的按 `面积p90 × 覆盖时长 × 居中度 × 人脸可用率` 排序。给了参考图就跳过打分，直接按相似度锁定。

**5. 取景路径：零相位低通 → RDP 关键帧 → PCHIP → HOLD → 非因果软限幅。**
因果滤波（One-Euro）会把抖动放大——它只压慢变化，不管高频。RDP 只保证折线贴合，Hermite 在拐点会拱出十几像素，所以补点迭代把误差压到 1.5 像素内。普通三次样条过冲 6–16px，PCHIP 是 0.00px。

---

## 性能（4K AV1，2583 帧 / 43 秒，RTX 4060 Ti）

| 环节 | 做法 | 速率 |
|---|---|---|
| 解码 | ffmpeg 管道（不是 OpenCV） | 4K 37.7 FPS / 1280 162 FPS / OpenCV 只有 22.6 |
| 检测 | 管道缩到 1280 + YOLO11m | 38–46 FPS |
| 人脸 | 裁头部（不放大过头）+ YuNet + SFace | 约 10 头/秒（CPU 单核瓶颈） |
| 渲染 | ffmpeg sendcmd 内部裁剪 + nvenc | 118–123 FPS |

端到端 2583 帧约 **4–7 分钟**（人脸占 70% 以上，是唯一剩下的瓶颈）。

**试过但不成立的加速**（别再走一遍）：

- YuNet 换 onnxruntime-GPU：单独 3.0ms（CPU 33.7ms），但放进 `embed_head` 反而慢一倍。每次调用都要 640 缩放 + canvas + NMS，全在 CPU，cv2 的实现在 C++ 里。
- SFace 换 GPU：单独快 4.6 倍、特征完全一致，但全流程慢 5 倍——和 torch 抢同一个 CUDA context，每次只传 112×112，传输比计算贵。（留了 `--face-gpu` 开关，默认关）
- 4K 原帧直接喂 YOLO：4.3 FPS。ultralytics 在 CPU 上把 4K 缩到 1280，必须先在管道里缩。

---

## 已知限制

1. **目标长时间不在画面里时，相机 HOLD 在原地。** 这是刻意的——真实拍摄里"停住等她自己走回来"比"甩镜头找别人"好。但超过 20 秒的静止画面观感确实差。
2. **人脸是唯一可靠模态。** 背对/低头时人脸失效，只能靠框的连续性撑 1.5 秒。人体 ReID 在这批素材上区分度不够（同一人 0.76–0.85，不同人也能到 0.67）。
3. **`--render-engine ffmpeg` 的时间戳对齐到帧。** 中位 PSNR 36.8dB，快速摇镜的个别帧 29dB（约 1–2px 位置差）。要像素级精确用 `--render-engine python`，慢约 4 倍。
4. **AV1 直接走管道解，不需要转码。** 但如果哪天遇到管道解不动的编码，`_ensure_proxy()` 会自动转一份 H.264。

---

## 目录

```
run.py                    主入口（一键批处理）
一键出片.bat              Windows 双击入口
ae/VVERTIGO_Track.jsx     AE 插件
configs/default.yaml      默认参数

core/
  fastvideo.py            ffmpeg 管道读取（比 OpenCV 快 1.7–7 倍）
  video.py                probe / OpenCV 解码（旧链路）
  schema.py               数据结构
  config.py               配置加载
pipeline/
  detect_fast.py          只要框的快速检测（管道 + 批量 YOLO）
  analyze.py              旧链路：切镜 + 跟踪
  identify.py             旧链路：身份聚类
  target.py, plan.py      旧链路的选人与规划
identity/
  face_anchor.py          主路径：人脸锚点 + 局部框关联
  face.py                 YuNet + SFace（cv2，CPU）
  face_onnx.py            onnxruntime 版（GPU，默认不启用）
  manager.py              旧链路的聚类与轨迹管理
camera/
  path.py                 低通 / RDP / PCHIP / 冻结 / 软限幅
  planner.py              逐帧规划取景
  framing.py              裁切框尺寸与位置
export/
  render.py               多路渲染 + ffmpeg 内部裁剪
  points.py               点位 CSV 导出（含误差自检）
tools/                    辅助脚本（验证、测量、对比）
tools/bench/              性能基准（解码 / 检测 / 渲染）
work/models/              yolo11m.pt, yunet.onnx, sface.onnx
待处理/                    把视频放这里
outputs/                  产物
```

---

## 环境

- **Python 3.12**（用 `py -3.12`，默认的 `python` 可能是别的版本）
- ffmpeg / ffprobe 在 PATH 里
- GPU：任意支持 CUDA 的 N 卡（YOLO + nvenc）；没 GPU 也能跑，慢几倍

```powershell
py -3.12 -m pip install -r requirements.txt
```

`torch` 请按自己的 CUDA 版本单独装（本机是 `cu124`）：

```powershell
py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## 旧链路说明

`--tracker taq / botsort` 是早期版本：做完整的切镜 + 跟踪 + 身份聚类，再选人。
它需要 `TAQ-MOTR-main` 目录（不在本项目内），而且 4K 上每条视频要跑一小时。

**现在不需要它。** 默认的 anchor 模式更快也更准。保留这些代码只是为了对照历史结果。
