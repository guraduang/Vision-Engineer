#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算模块 - 链码计算
计算轮廓的 Freeman 8 方向链码
"""

import cv2
import numpy as np
from typing import List, Tuple


def get_chain_code(contour: np.ndarray) -> List[int]:
    """
    计算轮廓的链码
    8方向链码：0=右, 1=右上, 2=上, 3=左上, 4=左, 5=左下, 6=下, 7=右下

    Args:
        contour: 轮廓点集

    Returns:
        8 方向链码列表 [0-7]
    """
    if len(contour) < 2:
        return []

    chain = []
    for i in range(len(contour)):
        pt1 = contour[i][0]
        pt2 = contour[(i + 1) % len(contour)][0]

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]

        # 计算方向
        if dx == 0 and dy == 0:
            continue

        angle = np.arctan2(-dy, dx)  # OpenCV y轴向下
        angle_deg = np.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360

        # 映射到8方向
        direction = int((angle_deg + 22.5) / 45) % 8
        chain.append(direction)

    return chain


def normalize_chain_code(chain: List[int]) -> List[int]:
    """
    归一化链码（旋转不变性）
    找到字典序最小的旋转

    Args:
        chain: 原始链码

    Returns:
        归一化后的链码
    """
    if not chain:
        return []

    n = len(chain)
    min_chain = chain

    for start in range(n):
        rotated = chain[start:] + chain[:start]
        if rotated < min_chain:
            min_chain = rotated

    return min_chain


def simplify_chain_code(chain: List[int], merge_threshold: int = 3) -> List[Tuple[int, int]]:
    """
    简化链码：合并连续相同方向

    Args:
        chain: 链码
        merge_threshold: 合并阈值，默认 3

    Returns:
        简化后的链码 [(direction, count), ...]
    """
    if not chain:
        return []

    simplified = []
    current_dir = chain[0]
    count = 1

    for i in range(1, len(chain)):
        if chain[i] == current_dir:
            count += 1
        else:
            if count >= merge_threshold:
                simplified.append((current_dir, count))
            current_dir = chain[i]
            count = 1

    if count >= merge_threshold:
        simplified.append((current_dir, count))

    return simplified
