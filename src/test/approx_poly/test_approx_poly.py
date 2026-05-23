#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试 approxPolyDP 多边形近似方法检测 L 型灯条角点
对比角度梯度法与多边形近似法的效果
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


def test_approx_poly(mask_path: str, output_dir: str = "output/approx_poly_test"):
    """测试 approxPolyDP 多边形近似方法"""
    os.makedirs(output_dir, exist_ok=True)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"错误: 无法读取图像 {mask_path}")
        return

    print(f"测试图像: {mask_path}")
    print(f"图像尺寸: {mask.shape}\n")

    # 形态学平滑
    kernel = np.ones((5, 5), np.uint8)
    mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours = find_contours(mask_smooth)
    if not contours:
        print("未检测到轮廓")
        return

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    print(f"轮廓点数: {len(contour)}, 周长: {perimeter:.1f}\n")

    # 测试不同 epsilon 值
    epsilon_ratios = [0.005, 0.01, 0.02, 0.03, 0.05]
    approx_results = []

    print("approxPolyDP 结果:")
    print(f"{'epsilon 比例':<14} {'epsilon 值':<12} {'顶点数'}")
    print("-" * 40)
    for ratio in epsilon_ratios:
        eps = ratio * perimeter
        approx = cv2.approxPolyDP(contour, eps, True)
        print(f"{ratio:<14.3f} {eps:<12.1f} {len(approx)}")
        approx_results.append((ratio, eps, approx))

    print()

    # 角度梯度法（对比）
    M = cv2.moments(contour)
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    angles = compute_contour_angles(contour, (cx, cy))
    gradient = compute_angle_gradient(angles)
    segments = detect_monotonic_segments(gradient, threshold=5.0, min_length=5)
    print(f"角度梯度法 (threshold=5.0): {len(segments)} 个单调区间\n")

    # 可视化
    n_approx = len(approx_results)
    fig, axes = plt.subplots(2, n_approx, figsize=(4 * n_approx, 8))

    points = contour.reshape(-1, 2)

    for i, (ratio, eps, approx) in enumerate(approx_results):
        verts = approx.reshape(-1, 2)

        # 上行: 轮廓 + 近似多边形顶点
        ax_top = axes[0, i]
        ax_top.plot(points[:, 0], points[:, 1], 'b-', linewidth=1, alpha=0.5, label='轮廓')
        ax_top.plot(
            np.append(verts[:, 0], verts[0, 0]),
            np.append(verts[:, 1], verts[0, 1]),
            'g-', linewidth=2, label='近似多边形'
        )
        ax_top.scatter(verts[:, 0], verts[:, 1], c='red', s=80, zorder=5, label='顶点')
        for j, (vx, vy) in enumerate(verts):
            ax_top.annotate(str(j + 1), (vx, vy), textcoords="offset points",
                            xytext=(5, 5), fontsize=9, color='red')
        ax_top.set_title(f'epsilon={ratio:.3f}\n顶点数: {len(verts)}')
        ax_top.axis('equal')
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(fontsize=7)

        # 下行: 在掩码图上标注顶点
        ax_bot = axes[1, i]
        ax_bot.imshow(mask_smooth, cmap='gray')
        ax_bot.scatter(verts[:, 0], verts[:, 1], c='red', s=80, zorder=5)
        poly_closed = np.append(verts, [verts[0]], axis=0)
        ax_bot.plot(poly_closed[:, 0], poly_closed[:, 1], 'g-', linewidth=2)
        for j, (vx, vy) in enumerate(verts):
            ax_bot.annotate(str(j + 1), (vx, vy), textcoords="offset points",
                            xytext=(5, 5), fontsize=9, color='yellow')
        ax_bot.set_title(f'掩码上的近似多边形')
        ax_bot.axis('off')

    plt.suptitle('approxPolyDP 不同 epsilon 值对比\n(L型灯条应有 6 个顶点)', fontsize=13)
    plt.tight_layout()

    basename = os.path.basename(mask_path).split('.')[0]
    output_path = f"{output_dir}/{basename}_approx_poly.jpg"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"可视化已保存: {output_path}")

    # 最优结果详细分析
    best_ratio, best_eps, best_approx = approx_results[2]  # epsilon=0.02
    verts = best_approx.reshape(-1, 2)
    print(f"\n最优结果 (epsilon={best_ratio}):")
    print(f"检测到 {len(verts)} 个顶点:")
    for j, (vx, vy) in enumerate(verts):
        print(f"  顶点 {j+1}: ({vx}, {vy})")

    # 计算各顶点间的角度
    print("\n各边方向角:")
    for j in range(len(verts)):
        p1 = verts[j]
        p2 = verts[(j + 1) % len(verts)]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        angle = np.degrees(np.arctan2(dy, dx))
        length = np.sqrt(dx**2 + dy**2)
        print(f"  边 {j+1}->{(j+1)%len(verts)+1}: 角度={angle:.1f}°, 长度={length:.1f}px")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 test_approx_poly.py <mask_path>")
        sys.exit(1)

    test_approx_poly(sys.argv[1])
