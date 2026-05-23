#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试不同参数对单调区间检测的影响
"""

import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from perception.contour_detector import find_contours
from calculation.contour_corner_detector import (
    compute_contour_angles,
    compute_angle_gradient,
    detect_monotonic_segments
)


def test_parameter_tuning(mask_path: str, output_dir: str = "output/parameter_tuning"):
    """测试不同参数组合"""
    os.makedirs(output_dir, exist_ok=True)

    # 读取掩码并平滑
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5, 5), np.uint8)
    mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 检测轮廓
    contours = find_contours(mask_smooth)
    contour = max(contours, key=cv2.contourArea)

    # 计算中心
    M = cv2.moments(contour)
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    print(f"轮廓点数: {len(contour)}\n")

    # 测试不同参数组合
    test_cases = [
        ("默认参数", 0.15, 3.0, 3.0, 10),
        ("小滤波窗口", 0.05, 3.0, 3.0, 10),
        ("大梯度阈值", 0.15, 3.0, 5.0, 10),
        ("更大梯度阈值", 0.15, 3.0, 8.0, 10),
        ("小窗口+大阈值", 0.05, 3.0, 5.0, 10),
        ("短区间", 0.15, 3.0, 3.0, 5),
    ]

    results = []

    for name, filter_ratio, sigma, grad_thresh, min_len in test_cases:
        print(f"{'='*60}")
        print(f"测试: {name}")
        print(f"  滤波窗口比例: {filter_ratio}")
        print(f"  高斯 sigma: {sigma}")
        print(f"  梯度阈值: {grad_thresh}")
        print(f"  最小区间长度: {min_len}")
        print(f"{'='*60}")

        # 计算角度
        angles = compute_contour_angles(contour, (cx, cy),
                                       filter_len_ratio=filter_ratio,
                                       sigma=sigma)
        gradient = compute_angle_gradient(angles)
        segments = detect_monotonic_segments(gradient,
                                            threshold=grad_thresh,
                                            min_length=min_len)

        angles_flat = angles.flatten()
        gradient_flat = gradient.flatten()

        print(f"角度范围: [{angles_flat.min():.1f}°, {angles_flat.max():.1f}°]")
        print(f"梯度范围: [{gradient_flat.min():.1f}, {gradient_flat.max():.1f}]")
        print(f"单调区间数: {len(segments)}")

        total_points_in_segments = 0
        for i, (start, end) in enumerate(segments):
            length = end - start + 1
            total_points_in_segments += length
            avg_angle = np.mean(angles_flat[start:end+1])
            print(f"  区间 {i+1}: [{start:4d}, {end:4d}], 长度 {length:3d}, 平均角度 {avg_angle:6.1f}°")

        coverage = total_points_in_segments / len(contour) * 100
        print(f"覆盖率: {coverage:.1f}% ({total_points_in_segments}/{len(contour)} 点)")
        print()

        results.append({
            'name': name,
            'angles': angles_flat,
            'gradient': gradient_flat,
            'segments': segments,
            'num_segments': len(segments),
            'coverage': coverage
        })

    # 创建对比可视化
    n_cases = len(results)
    fig, axes = plt.subplots(n_cases, 2, figsize=(16, 4*n_cases))

    if n_cases == 1:
        axes = axes.reshape(1, -1)

    for i, result in enumerate(results):
        # 子图 1: 角度曲线
        ax1 = axes[i, 0]
        ax1.plot(result['angles'], 'b-', linewidth=1)
        for start, end in result['segments']:
            ax1.axvspan(start, end, alpha=0.3, color='green')
        ax1.set_title(f"{result['name']}\nSegments: {result['num_segments']}, Coverage: {result['coverage']:.1f}%")
        ax1.set_xlabel('Point Index')
        ax1.set_ylabel('Angle (degrees)')
        ax1.grid(True, alpha=0.3)

        # 子图 2: 梯度曲线
        ax2 = axes[i, 1]
        ax2.plot(result['gradient'], 'r-', linewidth=1)
        ax2.axhline(y=3.0, color='g', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(y=-3.0, color='g', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(y=5.0, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(y=-5.0, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(y=8.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(y=-8.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        for start, end in result['segments']:
            ax2.axvspan(start, end, alpha=0.3, color='green')
        ax2.set_title('Gradient (Green zones = Monotonic segments)')
        ax2.set_xlabel('Point Index')
        ax2.set_ylabel('Gradient')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    basename = os.path.basename(mask_path).split('.')[0]
    output_path = f"{output_dir}/{basename}_parameter_comparison.jpg"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"{'='*60}")
    print(f"可视化已保存: {output_path}")
    print(f"{'='*60}\n")

    # 打印总结
    print("总结:")
    print(f"{'参数组合':<20} {'区间数':<8} {'覆盖率':<10} {'梯度范围'}")
    print("-" * 70)
    for r in results:
        grad_range = f"[{r['gradient'].min():.1f}, {r['gradient'].max():.1f}]"
        print(f"{r['name']:<20} {r['num_segments']:<8} {r['coverage']:>5.1f}%    {grad_range}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 test_parameter_tuning.py <mask_path>")
        sys.exit(1)

    mask_path = sys.argv[1]
    test_parameter_tuning(mask_path)
