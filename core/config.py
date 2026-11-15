# -*- coding: utf-8 -*-
"""配置加载：yaml + 点号路径覆盖（--set camera.max_speed=600）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = _coerce(value)


def _coerce(v: str):
    if not isinstance(v, str):
        return v
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v.startswith("[") and v.endswith("]"):
        return [float(x) for x in v[1:-1].split(",") if x.strip()]
    return v
