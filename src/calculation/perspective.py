#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算模块 - 透视矫正
透视变换将轮廓校正到正面视角
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def _order_quad_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """
    将最小外接矩形的四个角点排序为：左上、右上、右下、左下（图像坐标系，y 向下）。

    boxPoints 返回的顺序随矩形朝向变化，必须与目标矩形逐角对应后再 getPerspectiveTransform。
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def warp_to_frontal(image: np.ndarray,
                     contour: np.ndarray,
                     dst_size: int = 200,
                     padding: int = 10) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    透视变换：先求轮廓的最小旋转外接矩形（minAreaRect），以其四角为源四边形，
    再映射到带内边距的正方形目标区域，得到近似“正对”的矫正图。

    Args:
        image: 原图像
        contour: 轮廓
        dst_size: 输出图像边长（正方形）
        padding: 四周留白像素数，防止边缘截断

    Returns:
        (矫正后的图像, 矫正后的轮廓)，失败返回 (None, None)
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    width = int(rect[1][0])
    height = int(rect[1][1])

    if width == 0 or height == 0:
        return None, None

    # 源四点：最小旋转外接矩形角点，顺序为 TL, TR, BR, BL
    src_pts = _order_quad_tl_tr_br_bl(box)

    # 目标：与 src 同序 —— TL, TR, BR, BL；有效区域在 padding 内侧
    inner = dst_size - 2 * padding
    dst_pts = np.array([
        [padding, padding],
        [padding + inner, padding],
        [padding + inner, padding + inner],
        [padding, padding + inner],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (dst_size, dst_size))

    contour_reshaped = contour.reshape(-1, 1, 2).astype(np.float32)
    warped_contour = cv2.perspectiveTransform(contour_reshaped, M)

    return warped, warped_contour


def warp_to_frontal_with_matrix(
        image: np.ndarray,
        contour: np.ndarray,
        dst_size: int = 200,
        padding: int = 10) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    同 warp_to_frontal，额外返回 3×3 透视矩阵 M（原图 → warped）。

    Returns:
        (warped, warped_contour, M)，失败时 (None, None, None)
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    width = int(rect[1][0])
    height = int(rect[1][1])

    if width == 0 or height == 0:
        return None, None, None

    src_pts = _order_quad_tl_tr_br_bl(box)
    inner = dst_size - 2 * padding
    dst_pts = np.array([
        [padding, padding],
        [padding + inner, padding],
        [padding + inner, padding + inner],
        [padding, padding + inner],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (dst_size, dst_size))

    contour_reshaped = contour.reshape(-1, 1, 2).astype(np.float32)
    warped_contour = cv2.perspectiveTransform(contour_reshaped, M)

    return warped, warped_contour, M
