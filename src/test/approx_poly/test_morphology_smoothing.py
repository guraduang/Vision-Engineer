#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试形态学平滑对角度计算的影响
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


def test_morphology_smoothing(mask_path: str, output_dir: str = "output/morphology_test"):
    """测试形态学平滑效果"""
    os.makedirs(output_dir, exist_ok=True)

    # 读取二值化掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"错误: 无法读取图像 {mask_path}")
        return

    print(f"测试图像: {mask_path}")
    print(f"图像尺寸: {mask.shape}\n")

    # 测试不同的形态学操作
    test_cases = [
        ("原始", mask),
        ("开运算 3x3", cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))),
        ("开运算 5x5", cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))),
        ("闭运算 3x3", cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))),
        ("闭运算 5x5", cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))),
        ("开+闭 5x5", cv2.morphologyEx(
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)),
            cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )),
    ]

    results = []

    for name, processed_mask in test_cases:
        print(f"{'='*60}")
        print(f"测试: {name}")
        print(f"{'='*60}")

        # 检测轮廓
        contours = find_contours(processed_mask)
        if len(contours) == 0:
            print("未检测到轮廓\n")
            continue

        # 选择最大轮廓
        contour = max(contours, key=cv2.contourArea)
        print(f"轮廓点数: {len(contour)}")

        # 计算中心
        M = cv2.moments(contour)
        if M['m00'] == 0:
            print("轮廓面积为 0\n")
            continue

        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # 计算角度数组
        angles = compute_contour_angles(contour, (cx, cy))
        gradient = compute_angle_gradient(angles)
        segments = detect_monotonic_segments(gradient)

        angles_flat = angles.flatten()
        gradient_flat = gradient.flatten()

        print(f"角度范围: [{angles_flat.min():.1f}°, {angles_flat.max():.1f}°]")
        print(f"梯度范围: [{gradient_flat.min():.1f}, {gradient_flat.max():.1f}]")
        print(f"单调区间数: {len(segments)}")

        for i, (start, end) in enumerate(segments):
            length = end - start + 1
            print(f"  区间 {i+1}: [{start}, {end}], 长度 = {length}")

        results.append({
            'name': name,
            'mask': processed_mask,
            'contour': contour,
            'center': (cx, cy),
            'angles': angles_flat,
            'gradient': gradient_flat,
            'segments': segments,
            'num_points': len(contour),
            'num_segments': len(segments)
        })

        print()

    # 创建对比可视化
    n_cases = len(results)
    fig, axes = plt.subplots(n_cases, 4, figsize=(18, 4*n_cases))

    if n_cases == 1:
        axes = axes.reshape(1, -1)

    for i, result in enumerate(results):
        # 子图 1: 掩码
        ax1 = axes[i, 0]
        ax1.imshow(result['mask'], cmap='gray')
        cx, cy = result['center']
        ax1.plot(cx, cy, 'ro', markersize=8)
        ax1.set_title(f"{result['name']}\nPoints: {result['num_points']}")
        ax1.axis('off')

        # 子图 2: 轮廓
        ax2 = axes[i, 1]
        points = result['contour'].reshape(-1, 2)
        ax2.plot(points[:, 0], points[:, 1], 'b-', linewidth=1)
        ax2.plot(cx, cy, 'ro', markersize=8)
        ax2.set_title('Contour')
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)

        # 子图 3: 角度曲线
        ax3 = axes[i, 2]
        ax3.plot(result['angles'], 'b-', linewidth=1)
        for start, end in result['segments']:
            ax3.axvspan(start, end, alpha=0.2, color='green')
        ax3.set_title(f'Angle Curve\nSegments: {result["num_segments"]}')
        ax3.set_xlabel('Point Index')
        ax3.set_ylabel('Angle (deg)')
        ax3.grid(True, alpha=0.3)

        # 子图 4: 梯度曲线
        ax4 = axes[i, 3]
        ax4.plot(result['gradient'], 'r-', linewidth=1)
        ax4.axhline(y=3.0, color='g', linestyle='--', linewidth=1)
        ax4.axhline(y=-3.0, color='g', linestyle='--', linewidth=1)
        for start, end in result['segments']:
            ax4.axvspan(start, end, alpha=0.2, color='green')
        ax4.set_title('Gradient')
        ax4.set_xlabel('Point Index')
        ax4.set_ylabel('Gradient')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    basename = os.path.basename(mask_path).split('.')[0]
    output_path = f"{output_dir}/{basename}_morphology_comparison.jpg"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\n{'='*60}")
    print(f"对比可视化已保存: {output_path}")
    print(f"{'='*60}\n")

    # 打印总结
    print("总结:")
    print(f"{'方法':<15} {'轮廓点数':<10} {'单调区间数':<12} {'梯度范围'}")
    print("-" * 60)
    for r in results:
        grad_range = f"[{r['gradient'].min():.1f}, {r['gradient'].max():.1f}]"
        print(f"{r['name']:<15} {r['num_points']:<10} {r['num_segments']:<12} {grad_range}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 test_morphology_smoothing.py <mask_path>")
        sys.exit(1)

    mask_path = sys.argv[1]
    test_morphology_smoothing(mask_path)
