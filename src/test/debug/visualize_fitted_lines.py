#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直线段拟合可视化工具
展示每条直线段拟合成真正的直线后的效果
"""

import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from perception import extract_red_mask, find_contours, filter_contours_by_area
from test.debug.debug_line_extraction import extract_line_segments


def fit_line_to_segment(contour, line):
    """
    将直线段拟合成真正的直线

    Args:
        contour: 轮廓点
        line: Line 对象

    Returns:
        (start_point, end_point) - 拟合直线的起点和终点
    """
    # 提取线段对应的轮廓点
    pts = contour.reshape(-1, 2)
    segment_pts = pts[line.start_idx:line.end_idx+1]

    # 使用最小二乘法拟合直线
    [vx, vy, x0, y0] = cv2.fitLine(segment_pts, cv2.DIST_L2, 0, 0.01, 0.01)

    # 计算直线的起点和终点（延伸到线段范围）
    # 找到线段的边界点
    x_min, y_min = segment_pts.min(axis=0)
    x_max, y_max = segment_pts.max(axis=0)

    # 计算直线在边界处的点
    # 直线方程: (x - x0) / vx = (y - y0) / vy
    t_min = -1000
    t_max = 1000

    # 计算参数 t 的范围
    if abs(vx[0]) > 0.001:
        t1 = (x_min - x0[0]) / vx[0]
        t2 = (x_max - x0[0]) / vx[0]
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))

    if abs(vy[0]) > 0.001:
        t1 = (y_min - y0[0]) / vy[0]
        t2 = (y_max - y0[0]) / vy[0]
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))

    # 计算起点和终点
    start_point = (int(x0[0] + t_min * vx[0]), int(y0[0] + t_min * vy[0]))
    end_point = (int(x0[0] + t_max * vx[0]), int(y0[0] + t_max * vy[0]))

    return start_point, end_point


def visualize_fitted_lines(image_path: str):
    """可视化拟合的直线段"""

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return

    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = f"output/debug_fitted_lines/{basename}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== 直线段拟合可视化: {basename} ===")
    print(f"输出目录: {output_dir}\n")

    # 提取红色mask
    mask = extract_red_mask(image)

    # 查找轮廓
    contours = find_contours(mask)
    candidates = filter_contours_by_area(contours, min_area=100, max_area=50000)

    print(f"候选轮廓数: {len(candidates)}\n")

    # L型轮廓索引（根据你的说明：2, 3, 11, 9 是 L 型）
    l_shape_indices = [1, 2, 10, 8]  # 索引从0开始，所以是 1,2,10,8

    # 创建总览图
    overview_img = image.copy()

    for idx in l_shape_indices:
        if idx >= len(candidates):
            continue

        contour = candidates[idx]
        candidate_idx = idx + 1

        print(f"{'='*60}")
        print(f"Candidate #{candidate_idx} (L型轮廓)")
        print(f"轮廓点数: {len(contour)}")
        print(f"{'='*60}")

        # 提取直线段
        angles, gradient, lines, _straight_thr = extract_line_segments(
            contour, gradient_threshold=3.0, min_length=10
        )

        if lines:
            print(f"✓ 检测到 {len(lines)} 条直线段")
        else:
            print(f"✗ 未检测到直线段")

        # 创建单独的可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. 原始轮廓
        ax1 = axes[0]
        contour_img = np.zeros_like(image)
        cv2.drawContours(contour_img, [contour], -1, (255, 255, 255), 2)
        ax1.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Candidate #{candidate_idx}: Original Contour\n{len(contour)} points')
        ax1.axis('off')

        # 2. 轮廓 + 直线段标注
        ax2 = axes[1]
        segment_img = image.copy()
        cv2.drawContours(segment_img, [contour], -1, (0, 255, 255), 2)

        if lines:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            for line_idx, line in enumerate(lines):
                color = colors[line_idx % len(colors)]
                pts = contour.reshape(-1, 2)
                segment_pts = pts[line.start_idx:line.end_idx+1]

                # 绘制线段点
                for pt in segment_pts:
                    cv2.circle(segment_img, tuple(pt), 3, color, -1)

                # 标注
                cv2.putText(segment_img, f'L{line_idx+1}',
                           (int(line.center_x), int(line.center_y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            print(f"  直线段详情:")
            for line_idx, line in enumerate(lines):
                print(f"    Line {line_idx+1}: {line.length}点, 角度={line.avg_angle:.1f}°")

        ax2.imshow(cv2.cvtColor(segment_img, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Line Segments Detected: {len(lines) if lines else 0}')
        ax2.axis('off')

        # 3. 拟合的直线
        ax3 = axes[2]
        fitted_img = image.copy()
        cv2.drawContours(fitted_img, [contour], -1, (128, 128, 128), 1)

        if lines:
            for line_idx, line in enumerate(lines):
                color = colors[line_idx % len(colors)]

                # 拟合直线
                start_pt, end_pt = fit_line_to_segment(contour, line)

                # 绘制拟合的直线
                cv2.line(fitted_img, start_pt, end_pt, color, 3)

                # 标注角度
                mid_x = (start_pt[0] + end_pt[0]) // 2
                mid_y = (start_pt[1] + end_pt[1]) // 2
                cv2.putText(fitted_img, f'{line.avg_angle:.0f}°',
                           (mid_x, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                print(f"    拟合直线 {line_idx+1}: ({start_pt[0]},{start_pt[1]}) -> ({end_pt[0]},{end_pt[1]})")

        ax3.imshow(cv2.cvtColor(fitted_img, cv2.COLOR_BGR2RGB))
        ax3.set_title(f'Fitted Lines (Least Squares)')
        ax3.axis('off')

        plt.suptitle(f'Candidate #{candidate_idx} - Line Segment Fitting',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(output_dir, f'candidate_{candidate_idx:02d}_fitted.jpg')
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()

        print(f"  可视化已保存: {os.path.basename(output_path)}\n")

        # 添加到总览图
        if lines:
            cv2.drawContours(overview_img, [contour], -1, (0, 255, 0), 2)
            for line_idx, line in enumerate(lines):
                color = colors[line_idx % len(colors)]
                start_pt, end_pt = fit_line_to_segment(contour, line)
                cv2.line(overview_img, start_pt, end_pt, color, 3)
        else:
            cv2.drawContours(overview_img, [contour], -1, (0, 165, 255), 2)

        # 标注编号
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(overview_img, f"#{candidate_idx}", (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 保存总览图
    cv2.imwrite(os.path.join(output_dir, "00_overview.jpg"), overview_img)

    print(f"{'='*60}")
    print(f"总结")
    print(f"{'='*60}")
    print(f"所有输出已保存到: {output_dir}")
    print(f"  - 00_overview.jpg: 所有L型轮廓的拟合直线总览")
    print(f"  - candidate_XX_fitted.jpg: 每个候选的详细拟合过程")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 visualize_fitted_lines.py <image_path>")
        sys.exit(1)

    visualize_fitted_lines(sys.argv[1])
