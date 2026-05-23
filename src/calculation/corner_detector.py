#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算模块 - 外拐点检测
基于凸包缺陷检测 L 型灯条的外拐点
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


def find_outer_corners(contour: np.ndarray,
                       defect_threshold: float = 0.05) -> List[Tuple[int, int]]:
    """
    检测轮廓的外拐点（凸包缺陷对应的外拐点）

    Args:
        contour: 输入轮廓
        defect_threshold: 缺陷深度阈值（相对于轮廓周长），默认 0.05

    Returns:
        外拐点列表 [(x, y), ...]
    """
    if len(contour) < 5:
        return []

    # 计算凸包
    hull = cv2.convexHull(contour, returnPoints=False)

    if len(hull) < 3:
        return []

    # 计算凸包缺陷
    try:
        defects = cv2.convexityDefects(contour, hull)
    except:
        return []

    if defects is None:
        return []

    # 计算轮廓周长作为参考
    perimeter = cv2.arcLength(contour, True)
    depth_threshold = perimeter * defect_threshold

    outer_corners = []

    for i in range(defects.shape[0]):
        start_idx, end_idx, farthest_idx, depth = defects[i, 0]

        # 深度足够大的缺陷
        if depth / 256.0 > depth_threshold:
            # 缺陷的起点和终点就是外拐点
            start_pt = tuple(contour[start_idx][0])
            end_pt = tuple(contour[end_idx][0])

            outer_corners.append(start_pt)
            outer_corners.append(end_pt)

    # 去重（距离小于 5 像素的点合并）
    if len(outer_corners) == 0:
        return []

    unique_corners = [outer_corners[0]]
    for corner in outer_corners[1:]:
        is_duplicate = False
        for existing in unique_corners:
            dist = np.linalg.norm(np.array(corner) - np.array(existing))
            if dist < 5:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_corners.append(corner)

    return unique_corners


def split_contour_by_corners(contour: np.ndarray,
                              corners: List[Tuple[int, int]]) -> List[np.ndarray]:
    """
    根据外拐点将轮廓分成多段

    Args:
        contour: 输入轮廓
        corners: 外拐点列表

    Returns:
        轮廓段列表
    """
    if len(corners) < 2:
        return [contour]

    # 找到每个角点在轮廓中的索引
    corner_indices = []
    for corner in corners:
        min_dist = float('inf')
        min_idx = 0
        for i, pt in enumerate(contour):
            dist = np.linalg.norm(pt[0] - np.array(corner))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        corner_indices.append(min_idx)

    # 按索引排序
    corner_indices.sort()

    # 分割轮廓
    segments = []
    for i in range(len(corner_indices)):
        start_idx = corner_indices[i]
        end_idx = corner_indices[(i + 1) % len(corner_indices)]

        if end_idx > start_idx:
            segment = contour[start_idx:end_idx+1]
        else:
            # 跨越轮廓首尾
            segment = np.vstack([contour[start_idx:], contour[:end_idx+1]])

        if len(segment) > 1:
            segments.append(segment)

    return segments


def fit_line_hough(segment: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    对轮廓段进行霍夫直线拟合

    Args:
        segment: 轮廓段

    Returns:
        直线参数 (x1, y1, x2, y2) 或 None
    """
    if len(segment) < 2:
        return None

    # 使用 cv2.fitLine 拟合直线
    [vx, vy, x, y] = cv2.fitLine(segment, cv2.DIST_L2, 0, 0.01, 0.01)

    # 计算段的端点
    points = segment.reshape(-1, 2)

    # 将所有点投影到拟合直线上
    line_vec = np.array([vx[0], vy[0]])
    projections = []

    for pt in points:
        vec_to_pt = pt - np.array([x[0], y[0]])
        projection_length = np.dot(vec_to_pt, line_vec)
        projections.append(projection_length)

    # 找到投影的最小和最大值
    min_proj = min(projections)
    max_proj = max(projections)

    # 计算直线的两个端点
    x1 = int(x[0] + min_proj * vx[0])
    y1 = int(y[0] + min_proj * vy[0])
    x2 = int(x[0] + max_proj * vx[0])
    y2 = int(y[0] + max_proj * vy[0])

    return (x1, y1, x2, y2)


def find_line_intersection(line1: Tuple[float, float, float, float],
                            line2: Tuple[float, float, float, float]) -> Optional[Tuple[int, int]]:
    """
    计算两条直线的交点

    Args:
        line1: 直线1 (x1, y1, x2, y2)
        line2: 直线2 (x1, y1, x2, y2)

    Returns:
        交点 (x, y) 或 None
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # 计算直线方程的参数
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-6:
        return None  # 平行线

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    x = int(x1 + t * (x2 - x1))
    y = int(y1 + t * (y2 - y1))

    return (x, y)


def find_l_shape_keypoints(contour: np.ndarray) -> Optional[Dict]:
    """
    提取 L 型灯条的 3 个关键点（外 L 的三个顶点）
    算法：找到轮廓上离最小外接矩形 3 个角点最近的点（排除空缺角）

    Args:
        contour: 输入轮廓

    Returns:
        字典包含:
        - keypoints: 3 个关键点 [(x, y), ...]
        - lines: 2 条拟合直线 [(x1, y1, x2, y2), ...]
        或 None（如果检测失败）
    """
    if len(contour) < 10:
        return None

    # 1. 计算最小外接矩形
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 2. 对于矩形的 4 个角点，找到轮廓上离每个角点最近的点
    contour_points = contour.reshape(-1, 2)

    closest_points = []
    min_distances = []

    for corner in box:
        # 计算轮廓上所有点到这个矩形角点的距离
        distances = np.linalg.norm(contour_points - corner, axis=1)
        min_dist = np.min(distances)
        min_idx = np.argmin(distances)

        closest_points.append(tuple(contour_points[min_idx]))
        min_distances.append(min_dist)

    # 3. 找出距离最大的那个角点（这是空缺角，L 型不贴着这个角）
    max_dist_idx = np.argmax(min_distances)

    # 4. 选择其他 3 个角点对应的轮廓点作为关键点
    keypoints_original = []
    for i in range(4):
        if i != max_dist_idx:
            keypoints_original.append(closest_points[i])

    if len(keypoints_original) != 3:
        return None

    # 5. 找到拐点（夹角最接近 90 度的点）
    angles = []
    for i, kp in enumerate(keypoints_original):
        other_pts = [keypoints_original[j] for j in range(3) if j != i]
        vec1 = np.array(other_pts[0]) - np.array(kp)
        vec2 = np.array(other_pts[1]) - np.array(kp)

        # 计算夹角
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angles.append(angle)

    # 拐点应该是夹角最接近 90 度的点
    corner_idx = np.argmin([abs(a - np.pi/2) for a in angles])

    hull_pts = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
    perimeter = cv2.arcLength(contour, True)
    hull_tol = max(6.0, 0.01 * float(perimeter))

    def _dist_to_hull(pt: Tuple[int, int]) -> float:
        p = np.array(pt, dtype=np.float64)
        return float(np.min(np.linalg.norm(hull_pts - p, axis=1)))

    # 强透视或边缘噪声下，minAreaRect 近点可能落到 L 的内凹口。
    # 外拐点应贴近凸包；若 90° 候选明显不在凸包上，则在凸包候选中重选。
    if _dist_to_hull(keypoints_original[corner_idx]) > hull_tol:
        hull_like = [
            i for i, pt in enumerate(keypoints_original)
            if _dist_to_hull(pt) <= hull_tol
        ]
        if hull_like:
            corner_idx = min(hull_like, key=lambda i: abs(angles[i] - np.pi / 2))

    corner_pt = keypoints_original[corner_idx]

    # 另外两个点是端点
    endpoint_indices = [i for i in range(3) if i != corner_idx]
    endpoint1 = keypoints_original[endpoint_indices[0]]
    endpoint2 = keypoints_original[endpoint_indices[1]]

    # 重新排序关键点：端点1 -> 拐点 -> 端点2
    keypoints = [endpoint1, corner_pt, endpoint2]

    # 6. 拟合两条直线
    line1 = (endpoint1[0], endpoint1[1], corner_pt[0], corner_pt[1])
    line2 = (corner_pt[0], corner_pt[1], endpoint2[0], endpoint2[1])

    return {
        'keypoints': keypoints,
        'lines': [line1, line2],
        'contour': contour,
        'min_rect': box  # 添加最小外接矩形信息
    }
