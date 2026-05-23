#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算模块 - L 型匹配
基于链码和几何约束判断是否为 L 型
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict
from .chaincode import get_chain_code, normalize_chain_code, simplify_chain_code


def match_l_shape_chain(chain: List[int],
                         template: List[int] = [0, 2, 0, 6, 4, 2],
                         tolerance: int = 1) -> Tuple[bool, float]:
    """
    匹配 L 型链码模板

    Args:
        chain: 链码
        template: 标准 L 型链码，默认 [0, 2, 0, 6, 4, 2]
        tolerance: 允许的方向偏差，默认 1

    Returns:
        (是否匹配, 匹配分数)
    """
    if len(chain) < 4 or len(chain) > 8:
        return False, 0

    # 统计主方向（0,2,4,6）和对角方向（1,3,5,7）
    main_dirs = [d for d in chain if d in [0, 2, 4, 6]]
    diag_dirs = [d for d in chain if d in [1, 3, 5, 7]]

    # L型应该主要由主方向组成
    if len(main_dirs) < len(chain) * 0.5:  # 至少50%是主方向
        return False, 0

    # 检查是否包含至少3个不同的主方向
    unique_main = set(main_dirs)
    if len(unique_main) < 3:
        return False, 0

    # 计算匹配分数
    score = 0
    # 有4个主方向更好
    if len(unique_main) >= 4:
        score += 0.3
    elif len(unique_main) >= 3:
        score += 0.2

    # 长度合适
    if 5 <= len(chain) <= 7:
        score += 0.3
    elif 4 <= len(chain) <= 8:
        score += 0.2

    # 主方向占比高
    main_ratio = len(main_dirs) / len(chain)
    score += main_ratio * 0.4

    return score >= 0.5, score


def is_l_shape_chaincode(contour: np.ndarray,
                          image: np.ndarray,
                          epsilon_factor: float = 0.02) -> Tuple[bool, np.ndarray, Dict]:
    """
    使用链码判断是否为 L 型

    Args:
        contour: 轮廓
        image: 原图像
        epsilon_factor: 轮廓近似因子，默认 0.02

    Returns:
        (是否为 L 型, 近似轮廓, 详细信息字典)
    """
    # 轮廓近似
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 基本筛选
    if len(approx) < 4 or len(approx) > 8:
        return False, None, {}

    # 凸包比
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    convex_ratio = contour_area / hull_area if hull_area > 0 else 0

    if convex_ratio >= 0.85:  # L型应该是凹的
        return False, None, {}

    # 面积筛选
    area = cv2.contourArea(contour)
    if area < 100:
        return False, None, {}

    # 提取链码
    chain = get_chain_code(approx)
    if len(chain) < 4:
        return False, None, {}

    # 简化链码
    simplified = simplify_chain_code(chain, merge_threshold=1)
    simplified_dirs = [d for d, c in simplified]

    # 归一化
    normalized = normalize_chain_code(simplified_dirs)

    # 匹配L型模板
    is_match, match_ratio = match_l_shape_chain(normalized)

    # 检查边长约束
    lengths = []
    for i in range(len(approx)):
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % len(approx)][0]
        length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        lengths.append(length)

    # 寻找相似边对
    def edges_match(len1, len2, tolerance=0.20):  # 降低到20%
        ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        return ratio >= (1 - tolerance)

    pair_matches = 0
    matched_pairs = []
    for i in range(len(lengths)):
        for j in range(i + 1, len(lengths)):
            if edges_match(lengths[i], lengths[j]):
                pair_matches += 1
                matched_pairs.append((i, j, lengths[i], lengths[j]))

    info = {
        'approx': approx,
        'chain': chain,
        'simplified': simplified,
        'normalized': normalized,
        'match_ratio': match_ratio,
        'convex_ratio': convex_ratio,
        'area': area,
        'lengths': lengths,
        'pair_matches': pair_matches,
        'matched_pairs': matched_pairs
    }

    # 综合判断
    if is_match and pair_matches >= 2:
        return True, approx, info

    return False, None, info
