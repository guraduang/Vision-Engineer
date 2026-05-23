#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试不同配置对角点检测效果的影响
"""

import cv2
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.calculation.contour_corner_detector import extract_l_shape_corners
from src.perception.red_extractor import extract_red_mask
from src.utils.config_loader import get_config


def warp_l_shape_to_square(image: np.ndarray, contour: np.ndarray, square_size: int = 300):
    """
    将 L 型通过最小外接矩形透视变换成正方形

    Args:
        image: 原始图像
        contour: 轮廓
        square_size: 输出正方形的边长

    Returns:
        tuple: (变换后的图像, 透视变换矩阵)
    """
    # 计算最小外接矩形
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.float32(box)

    # 计算矩形的宽高
    width = int(rect[1][0])
    height = int(rect[1][1])

    # 确保宽度是较长边
    if width < height:
        width, height = height, width

    # 目标正方形的四个角点
    dst_pts = np.float32([
        [0, 0],
        [square_size, 0],
        [square_size, square_size],
        [0, square_size]
    ])

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(box, dst_pts)

    # 执行透视变换
    warped = cv2.warpPerspective(image, M, (square_size, square_size))

    return warped, M


def test_config_profiles(image_path: str):
    """
    测试不同配置文件对角点检测的影响

    Args:
        image_path: 测试图像路径
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 提取红色掩码
    mask = extract_red_mask(image)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("未找到轮廓")
        return

    # 按面积排序，选择最大的轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 配置文件列表
    profiles = ['default', 'high_precision', 'fast', 'experimental']

    print(f"\n测试图像: {os.path.basename(image_path)}")
    print("=" * 80)

    config = get_config()

    for profile in profiles:
        print(f"\n配置: {profile}")
        print("-" * 80)

        # 切换配置
        config.set_profile(profile)

        # 显示当前配置参数
        print(f"  梯度阈值: {config.get('gradient.threshold')}")
        print(f"  高斯滤波标准差: {config.get('angle_computation.gaussian_sigma')}")
        print(f"  最小角点间隔比例: {config.get('clustering.min_interval_ratio')}")
        print(f"  最小区间数要求: {config.get('algorithm.min_segments_required')}")

        # 统计结果
        new_count = 0
        fallback_count = 0
        failed_count = 0

        for i, contour in enumerate(contours[:10]):  # 测试前10个轮廓
            # 透视变换
            warped, M = warp_l_shape_to_square(image, contour)
            if warped is None:
                continue

            # 在变换后的图像中重新提取轮廓
            warped_mask = extract_red_mask(warped)
            warped_contours, _ = cv2.findContours(warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(warped_contours) == 0:
                continue

            warped_contour = max(warped_contours, key=cv2.contourArea)

            # 提取角点
            result = extract_l_shape_corners(warped_contour, warped)

            if result is None:
                failed_count += 1
            elif result['method'] == 'new':
                new_count += 1
            elif result['method'] == 'fallback':
                fallback_count += 1

        total = new_count + fallback_count + failed_count
        if total > 0:
            print(f"\n  结果统计 (共 {total} 个轮廓):")
            print(f"    新算法成功: {new_count} ({new_count/total*100:.1f}%)")
            print(f"    降级到旧算法: {fallback_count} ({fallback_count/total*100:.1f}%)")
            print(f"    检测失败: {failed_count} ({failed_count/total*100:.1f}%)")
            print(f"    总成功率: {(new_count+fallback_count)/total*100:.1f}%")


def compare_profiles_on_dataset():
    """
    在整个数据集上比较不同配置的效果
    """
    test_images = ['data/1.png', 'data/2.png', 'data/3.png']

    for image_name in test_images:
        image_path = os.path.join('/home/workspace/vision', image_name)
        if os.path.exists(image_path):
            test_config_profiles(image_path)
        else:
            print(f"图像不存在: {image_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 测试单个图像
        test_config_profiles(sys.argv[1])
    else:
        # 测试整个数据集
        compare_profiles_on_dataset()
