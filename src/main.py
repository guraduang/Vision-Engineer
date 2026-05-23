#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主程序入口 - L 型灯条检测
预处理与 L 型链码识别逻辑统一以 debug_line_extraction.py 为准。

汇总图 `02_all_candidates.jpg` 仅绘制角点六边提取成功的轮廓，同色显示，不展示被筛除或失败的候选。
"""

import importlib.util
import os
import sys

_DBG_PATH = os.path.join(os.path.dirname(__file__), "test", "debug", "debug_line_extraction.py")
_spec = importlib.util.spec_from_file_location("debug_line_extraction", _DBG_PATH)
dle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dle)


def process_image(image_path: str):
    """
    处理单张图像。
    直接复用 debug_line_extraction 主链路，确保 main 与调试逻辑完全一致。
    """
    dle.debug_line_extraction(image_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    process_image(image_path)
