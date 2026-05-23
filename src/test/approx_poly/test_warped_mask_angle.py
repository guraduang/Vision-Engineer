#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试单个透视变换后的二值化灯条的角度计算
"""

import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from perception.contour_detector import find_contours
from calculation.contour_corner_detector import (
    compute_contour_angles,
    compute_angle_gradient,
    detect_monotonic_segments
)


def test_warped_mask(mask_path: str, output_dir: str = "output/warped_angle_test"):
    """测试透视变换后的二值化掩码"""
    os.makedirs(output_dir, exist_ok=True)

    # 读取二值化掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"错误: 无法读取图像 {mask_path}")
        return

    print(f"测试图像: {mask_path}")
    print(f"图像尺寸: {mask.shape}")

    # 检测轮廓
    contours = find_contours(mask)
    print(f"检测到 {len(contours)} 个轮廓")

    if len(contours) == 0:
        print("未检测到轮廓")
        return

    # 选择最大轮廓
    contour = max(contours, key=cv2.contourArea)
    print(f"最大轮廓点数: {len(contour)}")

    # 计算中心
    M = cv2.moments(contour)
    if M['m00'] == 0:
        print("轮廓面积为 0")
        return

    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    print(f"轮廓中心: ({cx:.1f}, {cy:.1f})")

    # 计算角度数组
    angles = compute_contour_angles(contour, (cx, cy))
    gradient = compute_angle_gradient(angles)
    segments = detect_monotonic_segments(gradient)

    angles_flat = angles.flatten()
    gradient_flat = gradient.flatten()

    print(f"角度范围: [{angles_flat.min():.1f}°, {angles_flat.max():.1f}°]")
    print(f"梯度范围: [{gradient_flat.min():.1f}, {gradient_flat.max():.1f}]")
    print(f"单调区间数: {len(segments)}")

    # 打印单调区间详情
    for i, (start, end) in enumerate(segments):
        length = end - start + 1
        print(f"  区间 {i+1}: [{start}, {end}], 长度 = {length}")

    # 创建可视化
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # 子图 1: 原始掩码
    ax1 = axes[0]
    ax1.imshow(mask, cmap='gray')
    ax1.plot(cx, cy, 'ro', markersize=10)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Binary Mask')
    ax1.axis('equal')

    # 子图 2: 轮廓形状
    ax2 = axes[1]
    points = contour.reshape(-1, 2)
    ax2.plot(points[:, 0], points[:, 1], 'b-', linewidth=2)
    ax2.plot(cx, cy, 'ro', markersize=10, label='Center')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Contour Shape')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # 子图 3: 角度曲线
    ax3 = axes[2]
    ax3.plot(angles_flat, 'b-', linewidth=1.5, label='Angle')
    ax3.set_xlabel('Contour Point Index')
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_title('Angle Curve (Smoothed)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 标记单调区间
    for start, end in segments:
        ax3.axvspan(start, end, alpha=0.2, color='green')

    # 子图 4: 梯度曲线
    ax4 = axes[3]
    ax4.plot(gradient_flat, 'r-', linewidth=1.5, label='Gradient')
    ax4.axhline(y=3.0, color='g', linestyle='--', label='Threshold (+3)')
    ax4.axhline(y=-3.0, color='g', linestyle='--', label='Threshold (-3)')
    ax4.set_xlabel('Contour Point Index')
    ax4.set_ylabel('Gradient')
    ax4.set_title('Angle Gradient')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 标记单调区间
    for start, end in segments:
        ax4.axvspan(start, end, alpha=0.2, color='green')

    plt.tight_layout()

    basename = os.path.basename(mask_path).split('.')[0]
    output_path = f"{output_dir}/{basename}_analysis.jpg"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\n可视化已保存: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 test_warped_mask_angle.py <mask_path>")
        sys.exit(1)

    mask_path = sys.argv[1]
    test_warped_mask(mask_path)
