# -*- coding: utf-8 -*-
"""统一数据契约。

所有模块只通过这里定义的几类对象交换数据：
检测器 -> TrackObs；跟踪器 -> TrackObs(local_id)；Identity -> IdentityObs；
Camera -> CameraPose。换模型、换跟踪器、换 ReID，只要仍然吐这几类对象，
下游（虚拟摄影机、渲染、评估）一行都不用改。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------- 视频与镜头
@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float
    has_audio: bool = False
    codec: str = ""


@dataclass
class Shot:
    shot_id: int
    start_frame: int          # 含
    end_frame: int            # 不含
    start_t: float
    end_t: float
    confidence: float = 1.0   # 切镜置信度，低于 confirm 阈值的标记为"疑似"

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    def contains(self, fi: int) -> bool:
        return self.start_frame <= fi < self.end_frame


# ---------------------------------------------------------------- 跟踪观测
@dataclass
class TrackObs:
    """一帧一个人。detection 与 tracking 的中间表示。"""
    fi: int
    t: float
    local_id: int             # 跟踪器给的局部 ID，切镜后会被复用
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 0.0        # 检测置信度（ultralytics 不提供关联置信度）
    shot_id: int = -1

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def foot(self) -> float:
        return self.y2

    @property
    def key(self) -> tuple[int, int]:
        return (self.shot_id, self.local_id)

    def to_dict(self) -> dict:
        return dict(fi=self.fi, t=round(self.t, 4), lid=self.local_id,
                    sid=self.shot_id, x1=round(self.x1, 1), y1=round(self.y1, 1),
                    x2=round(self.x2, 1), y2=round(self.y2, 1), s=round(self.score, 3))


# ---------------------------------------------------------------- 身份
@dataclass
class IdentityObs:
    """目标人物在某一帧的判定结果。这是摄影机唯一被允许读取的人物数据。"""
    fi: int
    t: float
    global_id: int
    local_id: Optional[int]
    x1: float
    y1: float
    x2: float
    y2: float
    appearance_score: float = 0.0   # 与身份库的相似度
    motion_score: float = 0.0       # 与位置预测的一致度
    identity_conf: float = 0.0      # 综合置信度 0-1
    decision: str = "UNKNOWN"       # BIND / RECOVER / NEW / REJECT
    shot_id: int = -1

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0


@dataclass
class IdentityEvent:
    t: float
    fi: int
    local_id: int
    global_id: int
    action: str                  # BIND / CONFLICT / SWAP_SUSPECT / NEW_IDENTITY
    appearance_score: float = 0.0
    motion_score: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- 摄影机
@dataclass
class CameraPose:
    """一帧的取景指令。这是渲染阶段唯一的输入。"""
    fi: int
    t: float
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    center_x: float
    target_id: int
    state: str                  # TRACKING / PREDICT / HOLD / RECOVER / LOST
    confidence: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["center_x"] = round(self.center_x, 1)
        d["confidence"] = round(self.confidence, 3)
        d["t"] = round(self.t, 4)
        return d


@dataclass
class FrameBox:
    """取景框（framing box）：由人体框派生，只用于构图，不用于身份。"""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


# ---------------------------------------------------------------- 文件 IO
def write_jsonl(path: str | Path, rows) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r if isinstance(r, dict) else r.to_dict(),
                               ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
