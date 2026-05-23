#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化角度计算过程
展示 compute_contour_angles 函数的工作原理
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

from perception.red_extractor import extract_red_mask
from perception.contour_detector import find_contours, filter_contours_by_area
from calculation.contour_corner_detector import (
    compute_contour_angles,
    compute_angle_gradient,
    detect_monotonic_segments
)


def visualize_angle_computation(image_path: str, output_dir: str = "output/angle_vis"):
    """可视化角度计算过程"""
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    mask = extract_red_mask(image)
    contours = find_contours(mask)
    contours = filter_contours_by_area(contours, min_area=200, max_area=10000)

    print(f"检测到 {len(contours)} 个轮廓")

    # 处理前 3 个轮廓
    for idx, contour in enumerate(contours[:3]):
        print(f"\n处理轮廓 #{idx+1}")

        # 计算中心
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # 计算角度数组
        angles = compute_contour_angles(contour, (cx, cy))
        gradient = compute_angle_gradient(angles)
        segments = detect_monotonic_segments(gradient)

        angles_flat = angles.flatten()
        gradient_flat = gradient.flatten()

        # 创建可视化图
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 子图 1: 轮廓形状
        ax1 = axes[0]
        points = contour.reshape(-1, 2)
        ax1.plot(points[:, 0], points[:, 1], 'b-', linewidth=2)
        ax1.plot(cx, cy, 'ro', markersize=10, label='中心点')
        ax1.set_xlabel('X 坐标')
        ax1.set_ylabel('Y 坐标')
        ax1.set_title(f'轮廓 #{idx+1} 形状')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # 子图 2: 角度曲线
        ax2 = axes[1]
        ax2.plot(angles_flat, 'b-', linewidth=1.5, label='角度曲线')
        ax2.set_xlabel('轮廓点索引')
        ax2.set_ylabel('角度 (度)')
        ax2.set_title('角度曲线（平滑后）')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 标记单调区间
        for start, end in segments:
            ax2.axvspan(start, end, alpha=0.2, color='green')

        # 子图 3: 梯度曲线
        ax3 = axes[2]
        ax3.plot(gradient_flat, 'r-', linewidth=1.5, label='梯度')
        ax3.axhline(y=3.0, color='g', linestyle='--', label='阈值 (+3)')
        ax3.axhline(y=-3.0, color='g', linestyle='--', label='阈值 (-3)')
        ax3.set_xlabel('轮廓点索引')
        ax3.set_ylabel('梯度')
        ax3.set_title('角度梯度（一阶差分）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 标记单调区间
        for start, end in segments:
            ax3.axvspan(start, end, alpha=0.2, color='green')

        plt.tight_layout()
        output_path = f"{output_dir}/contour_{idx+1}_analysis.jpg"
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"  角度范围: [{angles_flat.min():.1f}°, {angles_flat.max():.1f}°]")
        print(f"  梯度范围: [{gradient_flat.min():.1f}, {gradient_flat.max():.1f}]")
        print(f"  单调区间数: {len(segments)}")
        print(f"  可视化已保存: {output_path}")

    print(f"\n所有可视化已保存到: {output_dir}/")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 visualize_angle_computation.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    visualize_angle_computation(image_path)
