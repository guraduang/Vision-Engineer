#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算模块 - 基于轮廓角度分析的角点检测
移植自 RM 视觉核心的条状突起检测算法，适配 L 型灯条的凹陷特征检测
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config_loader import get_config


def compute_contour_angles(contour: np.ndarray, center: Tuple[float, float],
                          filter_len_ratio: Optional[float] = None,
                          sigma: Optional[float] = None) -> np.ndarray:
    """
    计算轮廓每个点相对于中心的角度，并使用高斯滤波平滑

    参考: rm_vision_core/rune_fan_active.cpp:getAngles()

    Args:
        contour: 输入轮廓 (N, 1, 2)
        center: 中心点 (cx, cy)
        filter_len_ratio: 高斯滤波窗口长度比例（None 则从配置读取）
        sigma: 高斯滤波标准差（None 则从配置读取）

    Returns:
        平滑后的角度数组 (1, N)，单位为度，处理了周期性跳变
    """
    # 从配置读取参数
    config = get_config()
    if filter_len_ratio is None:
        filter_len_ratio = config.get('angle_computation.filter_len_ratio', 0.15)
    if sigma is None:
        sigma = config.get('angle_computation.gaussian_sigma', 3.0)

    # 展平轮廓为 (N, 2)
    points = contour.reshape(-1, 2).astype(np.float32)
    n = len(points)

    # 计算方向向量（相邻点的差分）
    directions = np.zeros((2, n), dtype=np.float32)
    directions[:, :-1] = (points[1:] - points[:-1]).T
    directions[:, -1] = (points[0] - points[-1]).T  # 闭合轮廓

    # 使用高斯滤波平滑方向向量
    # 滤波器长度：至少17，最多101，取轮廓长度的一定比例
    filter_len = max(17, min(int(filter_len_ratio * n) | 1, 101))

    kernel = cv2.getGaussianKernel(filter_len, sigma, cv2.CV_32F).T
    kernel = kernel / np.sum(kernel)  # 归一化

    # 对 x 和 y 方向分别滤波
    directions[0] = cv2.filter2D(directions[0].reshape(1, -1), -1, kernel,
                                  borderType=cv2.BORDER_DEFAULT).flatten()
    directions[1] = cv2.filter2D(directions[1].reshape(1, -1), -1, kernel,
                                  borderType=cv2.BORDER_DEFAULT).flatten()

    # 计算角度（使用 atan2）
    angles = np.arctan2(directions[1], directions[0]) * 180.0 / np.pi

    # 处理角度周期性跳变（-180° 到 180° 的跳变）
    # 累积跳变次数，使角度连续
    n_wraps = 0
    last_angle = 0.0
    unwrapped_angles = np.zeros(n, dtype=np.float32)

    for i in range(n):
        angle = angles[i]
        if angle - last_angle < -180:
            n_wraps += 1
        elif angle - last_angle > 180:
            n_wraps -= 1
        last_angle = angle
        unwrapped_angles[i] = angle + n_wraps * 360.0

    return unwrapped_angles.reshape(1, -1)


def compute_angle_gradient(angles: np.ndarray) -> np.ndarray:
    """
    计算角度数组的梯度（一阶差分）

    Args:
        angles: 角度数组 (1, N)

    Returns:
        梯度数组 (1, N)
    """
    # 使用中心差分核 [-1, 0, 1]
    kernel = np.array([[-1, 0, 1]], dtype=np.float32)
    gradient = cv2.filter2D(angles, -1, kernel, borderType=cv2.BORDER_DEFAULT)

    return gradient


def detect_monotonic_segments(gradient: np.ndarray, threshold: Optional[float] = None,
                             min_length: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    检测梯度变化小的区间（单调区间，对应直线段）

    Args:
        gradient: 梯度数组 (1, N)
        threshold: 梯度阈值，小于此值认为是直线段（None 则从配置读取）
        min_length: 最小区间长度（None 则从配置读取）

    Returns:
        单调区间列表 [(start_idx, end_idx), ...]
    """
    # 从配置读取参数
    config = get_config()
    if threshold is None:
        threshold = config.get('gradient.threshold', 3.0)
    if min_length is None:
        min_length = config.get('gradient.min_segment_length', 10)

    gradient_flat = gradient.flatten()
    n = len(gradient_flat)

    segments = []
    in_segment = False
    start_idx = 0

    for i in range(n):
        grad = abs(gradient_flat[i])

        if grad <= threshold:
            if not in_segment:
                in_segment = True
                start_idx = i
        else:
            if in_segment:
                in_segment = False
                # 只保留长度足够的区间
                if i - 1 - start_idx >= min_length:
                    segments.append((start_idx, i - 1))

    # 处理最后一个区间
    if in_segment and n - 1 - start_idx >= min_length:
        segments.append((start_idx, n - 1))

    # 合并距离过近且角度相近的区间
    # 参考 RM 代码：如果两个区间间隔 < 20 且角度差 < 5°，则合并
    if len(segments) < 2:
        return segments

    merged_segments = [segments[0]]
    for i in range(1, len(segments)):
        prev_start, prev_end = merged_segments[-1]
        curr_start, curr_end = segments[i]

        # 检查间隔和角度差（这里简化处理，只检查间隔）
        if curr_start - prev_end < 20:
            # 合并区间
            merged_segments[-1] = (prev_start, curr_end)
        else:
            merged_segments.append(segments[i])

    return merged_segments


def cluster_corner_candidates(corners: List[int], contour: np.ndarray,
                               gradient: np.ndarray, min_interval: Optional[int] = None,
                               image_size: int = 300) -> List[int]:
    """
    邻点聚类压缩：合并距离过近的角点候选，保留曲率最大的点

    Args:
        corners: 角点候选索引列表
        contour: 轮廓 (N, 1, 2)
        gradient: 梯度数组 (1, N)
        min_interval: 最小间隔，小于此值的角点会被合并（None 则从配置读取）
        image_size: 图像尺寸（用于计算自适应间隔）

    Returns:
        聚类后的角点索引列表
    """
    if len(corners) <= 1:
        return corners

    # 从配置读取参数
    config = get_config()
    if min_interval is None:
        min_interval_ratio = config.get('clustering.min_interval_ratio', 0.05)
        min_interval = int(image_size * min_interval_ratio)

    gradient_flat = gradient.flatten()
    clustered = []

    i = 0
    while i < len(corners):
        cluster = [corners[i]]
        j = i + 1

        # 收集距离过近的角点
        while j < len(corners) and corners[j] - corners[i] < min_interval:
            cluster.append(corners[j])
            j += 1

        # 在聚类中选择梯度绝对值最大的点（曲率最大）
        best_idx = cluster[0]
        max_grad = abs(gradient_flat[best_idx])

        for idx in cluster[1:]:
            if abs(gradient_flat[idx]) > max_grad:
                max_grad = abs(gradient_flat[idx])
                best_idx = idx

        clustered.append(best_idx)
        i = j

    return clustered


def detect_inner_corner(contour: np.ndarray, outer_keypoints: List[Tuple[int, int]],
                        angles: np.ndarray, gradient: np.ndarray,
                        image_size: int = 300) -> Optional[Tuple[int, int]]:
    """
    检测内拐点（凹陷特征）

    L 型灯条的内拐点是一个凹陷，对应角度梯度的负峰值

    Args:
        contour: 轮廓 (N, 1, 2)
        outer_keypoints: 外轮廓的 3 个角点 [端点1, 拐点, 端点2]
        angles: 角度数组 (1, N)
        gradient: 梯度数组 (1, N)
        image_size: 图像尺寸

    Returns:
        内拐点坐标 (x, y)，如果找不到返回 None
    """
    if len(outer_keypoints) != 3:
        return None

    points = contour.reshape(-1, 2)
    gradient_flat = gradient.flatten()

    endpoint1, outer_corner, endpoint2 = outer_keypoints

    # 1. 找到外拐点在轮廓中的索引
    outer_corner_idx = None
    min_dist = float('inf')
    for i, pt in enumerate(points):
        dist = np.linalg.norm(pt - np.array(outer_corner))
        if dist < min_dist:
            min_dist = dist
            outer_corner_idx = i

    if outer_corner_idx is None:
        return None

    # 2. 在外拐点附近搜索梯度负峰值（凹陷特征）
    # 搜索范围：外拐点前后各 1/4 轮廓长度
    n = len(points)
    search_range = n // 4

    # 计算搜索区间
    start_idx = (outer_corner_idx - search_range) % n
    end_idx = (outer_corner_idx + search_range) % n

    # 收集搜索区间内的索引
    if start_idx < end_idx:
        search_indices = list(range(start_idx, end_idx + 1))
    else:
        search_indices = list(range(start_idx, n)) + list(range(0, end_idx + 1))

    # 3. 找到梯度最小值（最负）的点
    min_gradient = 0
    inner_corner_idx = None

    for idx in search_indices:
        if gradient_flat[idx] < min_gradient:
            min_gradient = gradient_flat[idx]
            inner_corner_idx = idx

    if inner_corner_idx is None:
        return None

    # 4. 验证几何约束：内拐点应该在 L 型内侧
    inner_pt = points[inner_corner_idx]

    # 计算外拐点到两个端点的向量
    vec1 = np.array(endpoint1) - np.array(outer_corner)
    vec2 = np.array(endpoint2) - np.array(outer_corner)

    # 计算外拐点到内拐点的向量
    vec_inner = inner_pt - np.array(outer_corner)

    # 内拐点应该在两个向量夹角的内侧
    # 使用叉积判断方向
    cross1 = np.cross(vec1, vec_inner)
    cross2 = np.cross(vec_inner, vec2)

    # 如果叉积同号，说明内拐点在夹角内侧
    if cross1 * cross2 > 0:
        return tuple(inner_pt)

    return None


def extract_l_shape_corners(contour: np.ndarray, image: np.ndarray,
                             image_size: int = 300) -> Optional[Dict]:
    """
    提取 L 型灯条的角点（混合算法：新算法 + 降级到旧算法）

    策略：
    1. 优先使用新算法（基于角度梯度分析）
    2. 当新算法失败时（单调区间 < 2），降级到旧算法（基于最小外接矩形）

    Args:
        contour: 输入轮廓 (N, 1, 2)
        image: 透视变换后的图像（用于验证）
        image_size: 图像尺寸

    Returns:
        字典包含:
        - outer_keypoints: 外轮廓 3 个角点 [(x, y), ...]
        - inner_corner: 内拐点 (x, y) 或 None
        - method: 'new' 或 'fallback'
        - angles: 角度数组（调试用，仅新算法）
        - gradient: 梯度数组（调试用，仅新算法）
        - segments: 单调区间列表（调试用，仅新算法）
        或 None（如果检测失败）
    """
    if len(contour) < 10:
        return None

    # 尝试新算法
    result_new = _extract_corners_new_algorithm(contour, image, image_size)

    if result_new is not None:
        result_new['method'] = 'new'
        return result_new

    # 降级到旧算法
    from src.calculation.corner_detector import find_l_shape_keypoints
    result_old = find_l_shape_keypoints(contour)

    if result_old is None:
        return None

    # 转换旧算法输出格式
    return {
        'outer_keypoints': result_old['keypoints'],
        'inner_corner': None,  # 旧算法不提供内拐点
        'method': 'fallback',
        'contour': contour
    }


def _monotonic_segment_corner_pipeline(
        contour: np.ndarray,
        image_size: int = 300) -> Optional[Dict]:
    """
    单调区间端点 → 聚类后的全部角点索引（不截断 top-3）。

    Returns:
        None 若轮廓无效、单调区间不足；否则 dict:
        angles, gradient, segments, corner_indices (List[int])
    """
    if len(contour) < 10:
        return None

    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    center = (cx, cy)

    angles = compute_contour_angles(contour, center)
    gradient = compute_angle_gradient(angles)
    segments = detect_monotonic_segments(gradient)

    config = get_config()
    min_segments_required = config.get('algorithm.min_segments_required', 2)
    if len(segments) < min_segments_required:
        return None

    corner_candidates: List[int] = []
    for start, end in segments:
        corner_candidates.append(start)
        corner_candidates.append(end)
    corner_candidates = sorted(list(set(corner_candidates)))

    corner_indices = cluster_corner_candidates(
        corner_candidates, contour, gradient, image_size=image_size)
    if not corner_indices:
        return None

    return {
        'angles': angles,
        'gradient': gradient,
        'segments': segments,
        'corner_indices': corner_indices,
    }


def get_all_segment_corners(
        contour: np.ndarray,
        image_size: int = 300) -> Optional[Dict]:
    """
    返回基于角度梯度单调区间的**全部**聚类角点（轮廓索引 + 像素坐标）。

    与 `extract_l_shape_corners` 中新算法的前半段一致，但不挑选 top-3，
    便于调试链码所需的完整边过渡点。

    Returns:
        None 若流水线不可用；否则:
        - corner_indices: List[int]
        - corner_points: List[Tuple[int, int]]
        - segments: 单调区间列表
    """
    pipe = _monotonic_segment_corner_pipeline(contour, image_size=image_size)
    if pipe is None:
        return None
    pts = contour.reshape(-1, 2)
    idxs = pipe['corner_indices']
    return {
        'corner_indices': list(idxs),
        'corner_points': [(int(pts[i][0]), int(pts[i][1])) for i in idxs],
        'segments': pipe['segments'],
        'angles': pipe['angles'],
        'gradient': pipe['gradient'],
    }


def _extract_corners_new_algorithm(contour: np.ndarray, image: np.ndarray,
                                    image_size: int = 300) -> Optional[Dict]:
    """
    新算法实现（基于角度梯度分析）

    移植自 RM 视觉核心的条状突起检测算法，适配 L 型灯条的凹陷特征检测

    Args:
        contour: 输入轮廓 (N, 1, 2)
        image: 透视变换后的图像（用于验证）
        image_size: 图像尺寸

    Returns:
        字典包含角点信息，或 None（如果检测失败）
    """
    if len(contour) < 10:
        return None

    pipe = _monotonic_segment_corner_pipeline(contour, image_size=image_size)
    if pipe is None:
        return None

    corners = pipe['corner_indices']
    angles = pipe['angles']
    gradient = pipe['gradient']
    segments = pipe['segments']

    if len(corners) < 3:
        return None

    # 7. 选择最显著的 3 个角点作为外轮廓角点
    # 根据梯度绝对值排序，选择前 3 个
    gradient_flat = gradient.flatten()
    corners_with_grad = [(idx, abs(gradient_flat[idx])) for idx in corners]
    corners_with_grad.sort(key=lambda x: x[1], reverse=True)

    top3_indices = [idx for idx, _ in corners_with_grad[:3]]
    top3_indices.sort()  # 按轮廓顺序排序

    # 8. 转换为坐标
    points = contour.reshape(-1, 2)
    outer_keypoints = [tuple(points[idx]) for idx in top3_indices]

    # 9. 识别拐点（夹角最接近 90 度的点）
    angles_between = []
    for i, kp in enumerate(outer_keypoints):
        other_pts = [outer_keypoints[j] for j in range(3) if j != i]
        vec1 = np.array(other_pts[0]) - np.array(kp)
        vec2 = np.array(other_pts[1]) - np.array(kp)

        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angles_between.append(angle)

    corner_idx = np.argmin([abs(a - np.pi/2) for a in angles_between])
    corner_pt = outer_keypoints[corner_idx]

    endpoint_indices = [i for i in range(3) if i != corner_idx]
    endpoint1 = outer_keypoints[endpoint_indices[0]]
    endpoint2 = outer_keypoints[endpoint_indices[1]]

    # 重新排序：端点1 -> 拐点 -> 端点2
    outer_keypoints_ordered = [endpoint1, corner_pt, endpoint2]

    # 10. 检测内拐点
    inner_corner = detect_inner_corner(contour, outer_keypoints_ordered, angles, gradient, image_size)

    return {
        'outer_keypoints': outer_keypoints_ordered,
        'inner_corner': inner_corner,
        'angles': angles,
        'gradient': gradient,
        'segments': segments,
        'contour': contour
    }
