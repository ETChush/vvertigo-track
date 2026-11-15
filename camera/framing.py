# -*- coding: utf-8 -*-
"""取景几何。

核心事实（决定整个设计）：1080p 素材上 9:16 的最大裁切是 608×1080——正好用满
整个画面高度。因此在「保持全身」这个前提下：

  * 纵向根本不需要跟随，也不可能切到脚（只要人在原始画面里）；
  * 唯一的自由度是水平平移；
  * 任何比 608 更窄的裁切都等于放大，而不是"让更多身体进画面"。

所以 zoom<1 的紧取景是用清晰度换构图，不是用画幅换全身。这是硬约束：1080p 源
不允许多要一个像素的高度。
"""
from __future__ import annotations

from core.schema import FrameBox


def crop_size(src_w: int, src_h: int, aspect: str = "9:16", use_full_height: bool = True,
              crop_width: int = 0, zoom: float = 1.0) -> tuple[int, int]:
    aw, ah = (int(x) for x in aspect.split(":"))
    if crop_width:
        cw = int(crop_width)
    elif use_full_height:
        cw = int(round(src_h * aw / ah))
    else:
        cw = int(round(src_h * aw / ah))
    cw = int(cw / zoom) // 2 * 2
    ch = int(round(cw * ah / aw)) // 2 * 2
    if ch > src_h:
        ch = src_h // 2 * 2
        cw = int(round(ch * aw / ah)) // 2 * 2
    cw = min(cw, src_w // 2 * 2)
    return cw, ch


def upscale_factor(cw: int, out_w: int = 1080) -> float:
    return out_w / max(cw, 1)


def framing_box(x1: float, y1: float, x2: float, y2: float,
                margin_x: float = 0.0, margin_top: float = 0.0) -> FrameBox:
    """由人体框生成取景框。第一版不加姿态关键点——纵向已满幅，加姿态没有收益。"""
    w, h = x2 - x1, y2 - y1
    return FrameBox(x1=x1 - w * margin_x, y1=y1 - h * margin_top,
                    x2=x2 + w * margin_x, y2=y2)


def crop_x_for(center_x: float, cw: int, src_w: int, margin: int = 0) -> int:
    left = center_x - cw / 2.0
    lo, hi = -margin, src_w - cw + margin
    left = max(lo, min(left, hi))
    return int(round(max(0, min(left, src_w - cw))))


def crop_y_center(y1: float, y2: float, ch: int, src_h: int,
                  headroom: float = 0.06) -> float:
    """纵向取景中心（返回的是裁切框中心的 y，不是顶边）。

    zoom=1 时 ch == src_h，纵向没有自由度（返回画面中心即可）。zoom>1 时裁切高度小于
    源高，纵向必须跟随，否则一定切头或切脚。

    规则：人物在框内垂直居中；当人体框比自己还高时（跳跃/近景），**优先保头**——
    头顶留 headroom 比例的边距，脚超出多少算多少。直拍成片里脸比脚重要。
    """
    bh = max(1e-6, y2 - y1)
    if ch >= src_h - 1:
        return src_h / 2.0
    if bh <= ch * (1.0 - 2 * headroom):
        return (y1 + y2) / 2.0
    # 保头：裁切框顶边 = y1 - ch*headroom
    return y1 - ch * headroom + ch / 2.0


def crop_y_for(center_y: float, ch: int, src_h: int) -> int:
    top = center_y - ch / 2.0
    return int(round(max(0.0, min(top, src_h - ch))))
