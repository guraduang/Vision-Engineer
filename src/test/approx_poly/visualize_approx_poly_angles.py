#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化 approxPolyDP 检测的顶点在角度梯度曲线上的位置
对比多边形近似法与角度梯度法
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


def find_vertex_indices(contour, vertices):
    """找到顶点在轮廓点序列中的索引"""
    contour_pts = contour.reshape(-1, 2)
    indices = []

    for vx, vy in vertices:
        # 找到距离最近的轮廓点
        distances = np.sqrt((contour_pts[:, 0] - vx)**2 + (contour_pts[:, 1] - vy)**2)
        idx = np.argmin(distances)
        indices.append(idx)

    return indices


def visualize_approx_poly_angles(mask_path: str, epsilon_ratio: float = 0.02,
                                   output_dir: str = "output/approx_poly_angles"):
    """可视化 approxPolyDP 顶点与角度梯度的关系"""
    os.makedirs(output_dir, exist_ok=True)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"错误: 无法读取图像 {mask_path}")
        return

    print(f"测试图像: {mask_path}")
    print(f"epsilon 比例: {epsilon_ratio}\n")

    # 形态学平滑
    kernel = np.ones((5, 5), np.uint8)
    mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours = find_contours(mask_smooth)
    if not contours:
        print("未检测到轮廓")
        return

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)

    # approxPolyDP 检测顶点
    epsilon = epsilon_ratio * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = approx.reshape(-1, 2)

    print(f"轮廓点数: {len(contour)}")
    print(f"检测到 {len(vertices)} 个顶点\n")

    # 计算角度和梯度
    M = cv2.moments(contour)
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    angles = compute_contour_angles(contour, (cx, cy), filter_len_ratio=0.15, sigma=3.0)
    gradient = compute_angle_gradient(angles)

    # 检测单调区间
    segments = detect_monotonic_segments(gradient, threshold=5.0, min_length=5)

    print(f"角度范围: [{np.min(angles):.1f}°, {np.max(angles):.1f}°]")
    print(f"梯度范围: [{np.min(gradient):.1f}, {np.max(gradient):.1f}]")
    print(f"检测到 {len(segments)} 个单调区间 (threshold=5.0)\n")

    # 找到顶点在轮廓中的索引
    vertex_indices = find_vertex_indices(contour, vertices)

    # 展平数组以便访问
    angles_flat = angles.flatten()
    gradient_flat = gradient.flatten()

    print("approxPolyDP 顶点对应的角度和梯度:")
    for i, (idx, (vx, vy)) in enumerate(zip(vertex_indices, vertices)):
        # 确保索引在有效范围内
        angle_val = angles_flat[idx] if idx < len(angles_flat) else 0.0
        grad_val = gradient_flat[idx] if idx < len(gradient_flat) else 0.0
        print(f"  顶点 {i+1}: 索引={idx:3d}, 坐标=({vx:3d}, {vy:3d}), "
              f"角度={angle_val:6.1f}°, 梯度={grad_val:6.1f}")

    # 可视化
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3)

    # 1. 轮廓 + approxPolyDP 顶点
    ax1 = fig.add_subplot(gs[0, 0])
    contour_pts = contour.reshape(-1, 2)
    ax1.plot(contour_pts[:, 0], contour_pts[:, 1], 'b-', linewidth=1, alpha=0.5, label='Original contour')
    poly_closed = np.append(vertices, [vertices[0]], axis=0)
    ax1.plot(poly_closed[:, 0], poly_closed[:, 1], 'g-', linewidth=2, label='approxPolyDP')
    ax1.scatter(vertices[:, 0], vertices[:, 1], c='red', s=100, zorder=5, label='Vertices')
    for i, (vx, vy) in enumerate(vertices):
        ax1.annotate(str(i + 1), (vx, vy), textcoords="offset points",
                     xytext=(8, 8), fontsize=10, color='red', fontweight='bold')
    ax1.set_title(f'Contour with {len(vertices)} approxPolyDP vertices (epsilon={epsilon_ratio})', fontsize=11)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # 2. 掩码上的顶点
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask_smooth, cmap='gray')
    ax2.scatter(vertices[:, 0], vertices[:, 1], c='red', s=100, zorder=5)
    ax2.plot(poly_closed[:, 0], poly_closed[:, 1], 'g-', linewidth=2)
    for i, (vx, vy) in enumerate(vertices):
        ax2.annotate(str(i + 1), (vx, vy), textcoords="offset points",
                     xytext=(8, 8), fontsize=10, color='yellow', fontweight='bold')
    ax2.set_title('Vertices on binary mask', fontsize=11)
    ax2.axis('off')

    # 3. 角度曲线 + 顶点位置标注
    ax3 = fig.add_subplot(gs[1, :])
    indices = np.arange(len(angles_flat))
    ax3.plot(indices, angles_flat, 'b-', linewidth=1.5, label='Angle curve')

    # 只绘制有效索引的顶点
    valid_vertex_indices = [idx for idx in vertex_indices if idx < len(angles_flat)]
    valid_vertex_angles = [angles_flat[idx] for idx in valid_vertex_indices]
    ax3.scatter(valid_vertex_indices, valid_vertex_angles, c='red', s=100, zorder=5, label='Vertices')

    for i, idx in enumerate(vertex_indices):
        if idx < len(angles_flat):
            ax3.axvline(idx, color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax3.annotate(f'V{i+1}', (idx, angles_flat[idx]), textcoords="offset points",
                         xytext=(0, 10), fontsize=9, color='red', ha='center', fontweight='bold')
    ax3.set_xlabel('Contour point index', fontsize=10)
    ax3.set_ylabel('Angle (degrees)', fontsize=10)
    ax3.set_title(f'Angle curve with {len(vertices)} approxPolyDP vertices marked', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # 4. 梯度曲线 + 顶点位置 + 单调区间
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(indices, gradient_flat, 'g-', linewidth=1.5, label='Gradient')
    ax4.axhline(5.0, color='orange', linestyle='--', linewidth=1, label='Threshold=5.0')
    ax4.axhline(-5.0, color='orange', linestyle='--', linewidth=1)

    # 只绘制有效索引的顶点
    valid_vertex_gradients = [gradient_flat[idx] for idx in valid_vertex_indices]
    ax4.scatter(valid_vertex_indices, valid_vertex_gradients, c='red', s=100, zorder=5, label='Vertices')

    # 标注单调区间
    for seg in segments:
        ax4.axvspan(seg[0], seg[1], alpha=0.2, color='cyan')

    for i, idx in enumerate(vertex_indices):
        if idx < len(gradient_flat):
            ax4.axvline(idx, color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax4.annotate(f'V{i+1}', (idx, gradient_flat[idx]), textcoords="offset points",
                         xytext=(0, 10), fontsize=9, color='red', ha='center', fontweight='bold')

    ax4.set_xlabel('Contour point index', fontsize=10)
    ax4.set_ylabel('Gradient', fontsize=10)
    ax4.set_title(f'Gradient curve: {len(segments)} monotonic segments (cyan) vs {len(vertices)} vertices (red)', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    plt.suptitle(f'approxPolyDP vs Angle Gradient Method\n'
                 f'approxPolyDP: {len(vertices)} vertices | Gradient method: {len(segments)} segments',
                 fontsize=13, fontweight='bold')

    basename = os.path.basename(mask_path).split('.')[0]
    output_path = f"{output_dir}/{basename}_angle_gradient.jpg"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可视化已保存: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 visualize_approx_poly_angles.py <mask_path> [epsilon_ratio]")
        sys.exit(1)

    mask_path = sys.argv[1]
    epsilon_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02

    visualize_approx_poly_angles(mask_path, epsilon_ratio)
