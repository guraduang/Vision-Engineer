#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
感知模块 - 轮廓检测
从二值掩码中检测和筛选轮廓
"""

import cv2
import numpy as np
from typing import Callable, List, Optional


def find_contours(mask: np.ndarray) -> List[np.ndarray]:
    """
    查找二值掩码中的轮廓

    Args:
        mask: 二值掩码

    Returns:
        轮廓列表
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours_by_area(contours: List[np.ndarray],
                              min_area: int = 100,
                              max_area: int = 50000) -> List[np.ndarray]:
    """
    按面积筛选轮廓

    Args:
        contours: 轮廓列表
        min_area: 最小面积，默认 100
        max_area: 最大面积，默认 50000

    Returns:
        筛选后的轮廓列表
    """
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered.append(contour)
    return filtered


def filter_contours_by_convexity(contours: List[np.ndarray],
                                   max_ratio: float = 0.85) -> List[np.ndarray]:
    """
    按凸包比筛选轮廓（L型应该是凹的）

    Args:
        contours: 轮廓列表
        max_ratio: 凸包比上限，默认 0.85

    Returns:
        筛选后的轮廓列表
    """
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            convexity_ratio = area / hull_area
            if convexity_ratio < max_ratio:
                filtered.append(contour)

    return filtered


def filter_contours_by_shape(contours: List[np.ndarray],
                              min_aspect_ratio: float = 1.5,
                              max_aspect_ratio: float = 6.0,
                              min_corners: int = 4,
                              max_corners: int = 8) -> List[np.ndarray]:
    """
    根据形状特征筛选轮廓（长宽比 + 角点数量）

    Args:
        contours: 轮廓列表
        min_aspect_ratio: 最小长宽比，默认 1.5
        max_aspect_ratio: 最大长宽比，默认 6.0
        min_corners: 最小角点数量，默认 4（宽容）
        max_corners: 最大角点数量，默认 8（宽容）

    Returns:
        筛选后的轮廓列表
    """
    filtered = []

    for contour in contours:
        # 1. 长宽比筛选
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]

        if width == 0 or height == 0:
            continue

        aspect_ratio = max(width, height) / min(width, height)

        if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
            continue

        # 2. 角点数量筛选（使用多边形逼近）
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_corners = len(approx)

        if not (min_corners <= num_corners <= max_corners):
            continue

        filtered.append(contour)

    return filtered


def filter_contours_by_geometry(contours: List[np.ndarray],
                                  max_side_ratio: float = 6.0,
                                  min_points: int = 7) -> List[np.ndarray]:
    """
    几何筛选：轮廓点数 + 最小外接矩形长宽比

    Args:
        max_side_ratio: 最大长宽比（长边/短边），排除过细长轮廓
        min_points: 最小轮廓点数，默认 7（即保留点数 > 6）
    """
    filtered = []
    for contour in contours:
        if len(contour) < min_points:
            continue

        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue

        side_ratio = max(w, h) / min(w, h)

        if side_ratio > max_side_ratio:
            continue

        filtered.append(contour)
    return filtered


def compute_warped_area_ratio(
        image: np.ndarray,
        contour: np.ndarray,
        warp_fn: Callable[[np.ndarray, np.ndarray], Optional[np.ndarray]],
        mask_fn: Callable[[np.ndarray], np.ndarray],
        warped_min_area: float = 50.0) -> Optional[float]:
    """
    计算单个候选轮廓在透视矫正后的面积占比

    Args:
        image: 原图像
        contour: 原图候选轮廓
        warp_fn: 透视矫正函数，接口 (image, contour) -> warped_image 或 None
        mask_fn: 掩码提取函数，接口 (warped_image) -> mask
        warped_min_area: warped 图上最小候选面积

    Returns:
        area_ratio: 最大候选轮廓面积 / warped 图总面积；失败返回 None
    """
    warped_image = warp_fn(image, contour)
    if warped_image is None:
        return None

    warped_mask = mask_fn(warped_image)
    warped_contours = find_contours(warped_mask)
    warped_candidates = filter_contours_by_area(
        warped_contours,
        min_area=warped_min_area,
        max_area=warped_image.shape[0] * warped_image.shape[1],
    )
    if not warped_candidates:
        return None

    main_contour = max(warped_candidates, key=cv2.contourArea)
    main_area = cv2.contourArea(main_contour)
    total_area = float(warped_image.shape[0] * warped_image.shape[1])
    if total_area <= 0:
        return None

    return main_area / total_area


def filter_contours_by_warped_area_ratio(
        image: np.ndarray,
        contours: List[np.ndarray],
        warp_fn: Callable[[np.ndarray, np.ndarray], Optional[np.ndarray]],
        mask_fn: Callable[[np.ndarray], np.ndarray],
        min_area_ratio: float = 0.10,
        max_area_ratio: float = 0.85,
        warped_min_area: float = 50.0) -> List[np.ndarray]:
    """
    第二层筛选：透视矫正后按面积占比过滤候选轮廓

    Args:
        image: 原图像
        contours: 第一层筛选后的候选轮廓
        warp_fn: 透视矫正函数，接口 (image, contour) -> warped_image 或 None
        mask_fn: 掩码提取函数，接口 (warped_image) -> mask
        min_area_ratio: 最小面积占比
        max_area_ratio: 最大面积占比
        warped_min_area: warped 图上最小候选面积

    Returns:
        通过透视面积占比筛选的轮廓列表
    """
    filtered = []
    for contour in contours:
        area_ratio = compute_warped_area_ratio(
            image=image,
            contour=contour,
            warp_fn=warp_fn,
            mask_fn=mask_fn,
            warped_min_area=warped_min_area,
        )
        if area_ratio is None:
            continue
        if min_area_ratio <= area_ratio <= max_area_ratio:
            filtered.append(contour)
    return filtered
