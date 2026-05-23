#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
感知模块 - 红色提取
从图像中提取红色区域
"""

import cv2
import numpy as np


def extract_red_mask(image: np.ndarray, threshold: int = 30) -> np.ndarray:
    """
    提取红色区域（R-B 差值法）

    Args:
        image: BGR 图像
        threshold: R-B 差值阈值，默认 30

    Returns:
        二值掩码（0/255）
    """
    b, g, r = cv2.split(image)
    red_diff = cv2.subtract(r, b)
    _, mask = cv2.threshold(red_diff, threshold, 255, cv2.THRESH_BINARY)

    return mask


def extract_red_mask_from_gray_contours(
        image: np.ndarray,
        gray_threshold: int = 20,
        red_diff_threshold: int = 30,
        red_ratio_threshold: float = 0.5,
        min_contour_area: float = 20.0) -> np.ndarray:
    """
    基于灰度轮廓 + 轮廓颜色判定提取红色区域

    处理流程：
    1. 对灰度图阈值化，先提取亮区轮廓
    2. 在每个轮廓内统计 R-B>阈值 的红色像素占比
    3. 仅保留红色占比达标的轮廓

    Args:
        image: BGR 图像
        gray_threshold: 灰度二值化阈值
        red_diff_threshold: 红色判定阈值（R-B > red_diff_threshold）
        red_ratio_threshold: 轮廓内红色像素占比阈值
        min_contour_area: 最小轮廓面积

    Returns:
        二值掩码（0/255）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, gray_binary = cv2.threshold(gray, gray_threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(gray_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    b, _, r = cv2.split(image)
    red_diff = cv2.subtract(r, b)

    mask = np.zeros(gray.shape, dtype=np.uint8)

    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue

        contour_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

        contour_pixels = contour_mask > 0
        total_pixels = int(np.count_nonzero(contour_pixels))
        if total_pixels == 0:
            continue

        red_pixels = np.count_nonzero(red_diff[contour_pixels] > red_diff_threshold)
        red_ratio = red_pixels / total_pixels

        if red_ratio >= red_ratio_threshold:
            cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    return mask
