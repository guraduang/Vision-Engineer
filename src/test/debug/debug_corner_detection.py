#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试工具 - 角点检测可视化

可视化角点检测的中间过程：
1. 角度曲线图
2. 梯度曲线图
3. 单调区间标注
4. 候选角点位置
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.red_extractor import extract_red_mask
from perception.contour_detector import find_contours, filter_contours_by_area
from calculation.contour_corner_detector import (
    compute_contour_angles,
    compute_angle_gradient,
    detect_monotonic_segments,
    extract_l_shape_corners
)


def visualize_corner_detection_debug(image_path: str, output_dir: str = "output/debug"):
    """
    可视化角点检测的中间过程

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 提取红色掩码
    mask = extract_red_mask(image)

    # 查找轮廓
    contours = find_contours(mask)
    contours = filter_contours_by_area(contours, min_area=200, max_area=10000)

    print(f"找到 {len(contours)} 个轮廓")

    # 对每个轮廓进行调试可视化
    for idx, contour in enumerate(contours):
        print(f"\n处理轮廓 {idx + 1}/{len(contours)}")

        # 计算中心
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        center = (cx, cy)

        # 计算角度和梯度
        angles = compute_contour_angles(contour, center)
        gradient = compute_angle_gradient(angles)

        # 检测单调区间
        segments = detect_monotonic_segments(gradient, threshold=3.0)

        # 绘制角度和梯度曲线
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # 子图1: 角度曲线
        axes[0].plot(angles.flatten(), 'b-', linewidth=1)
        axes[0].set_title(f'Contour {idx + 1}: Angle Curve')
        axes[0].set_xlabel('Point Index')
        axes[0].set_ylabel('Angle (degrees)')
        axes[0].grid(True, alpha=0.3)

        # 标注单调区间
        for start, end in segments:
            axes[0].axvspan(start, end, alpha=0.2, color='green')

        # 子图2: 梯度曲线
        axes[1].plot(gradient.flatten(), 'r-', linewidth=1)
        axes[1].axhline(y=3.0, color='g', linestyle='--', label='Threshold = 3.0')
        axes[1].axhline(y=-3.0, color='g', linestyle='--')
        axes[1].set_title(f'Contour {idx + 1}: Gradient Curve')
        axes[1].set_xlabel('Point Index')
        axes[1].set_ylabel('Gradient')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 标注单调区间
        for start, end in segments:
            axes[1].axvspan(start, end, alpha=0.2, color='green')

        # 子图3: 轮廓可视化
        vis_img = image.copy()
        cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 2)
        cv2.circle(vis_img, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        # 尝试检测角点
        result = extract_l_shape_corners(contour, image)

        if result is not None:
            # 绘制外角点
            for i, pt in enumerate(result['outer_keypoints']):
                cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
                cv2.putText(vis_img, f"{i+1}", (int(pt[0])+10, int(pt[1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 绘制内角点
            if result['inner_corner'] is not None:
                inner_pt = result['inner_corner']
                cv2.circle(vis_img, (int(inner_pt[0]), int(inner_pt[1])), 8, (255, 0, 255), -1)
                cv2.putText(vis_img, "Inner", (int(inner_pt[0])+10, int(inner_pt[1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # 显示使用的方法
            method = result.get('method', 'unknown')
            cv2.putText(vis_img, f"Method: {method}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            axes[2].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            axes[2].set_title(f'Contour {idx + 1}: Detected Corners (Method: {method})')
        else:
            axes[2].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            axes[2].set_title(f'Contour {idx + 1}: Detection Failed')

        axes[2].axis('off')

        # 保存图像
        plt.tight_layout()
        output_path = f"{output_dir}/contour_{idx + 1}_debug.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  - 单调区间数量: {len(segments)}")
        print(f"  - 检测结果: {'成功' if result is not None else '失败'}")
        if result is not None:
            print(f"  - 使用方法: {result.get('method', 'unknown')}")
        print(f"  - 调试图保存到: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 debug_corner_detection.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    visualize_corner_detection_debug(image_path)
