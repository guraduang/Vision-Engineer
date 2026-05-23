#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本 - 外拐点检测与 L 型关键点提取（分阶段可视化）
"""

import sys
import os
import cv2
import numpy as np

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from perception.red_extractor import extract_red_mask
from perception.contour_detector import find_contours, filter_contours_by_area
from calculation.corner_detector import find_l_shape_keypoints


def visualize_stages_for_contour(image, contour, contour_idx, output_dir="output/stages"):
    """
    分阶段可视化单个轮廓的处理过程

    Args:
        image: 原始图像
        contour: 轮廓
        contour_idx: 轮廓编号
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    contour_points = contour.reshape(-1, 2)

    # 阶段 1: 原始轮廓
    vis1 = image.copy()
    cv2.drawContours(vis1, [contour], -1, (0, 255, 0), 2)
    cv2.putText(vis1, f"Stage 1: Original Contour #{contour_idx}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(f"{output_dir}/contour_{contour_idx}_stage1_original.jpg", vis1)
    print(f"Stage 1 saved: 原始轮廓")

    # 阶段 2: 最小外接矩形 + 4 个角点
    vis2 = image.copy()
    cv2.drawContours(vis2, [contour], -1, (0, 255, 0), 2)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(vis2, [box], 0, (255, 0, 0), 2)

    # 标注矩形 4 个角点
    for i, corner in enumerate(box):
        cv2.circle(vis2, tuple(corner), 8, (0, 255, 255), -1)
        cv2.putText(vis2, f"C{i}", (corner[0]+8, corner[1]-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(vis2, f"Stage 2: Min Area Rect + 4 Corners", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imwrite(f"{output_dir}/contour_{contour_idx}_stage2_min_rect.jpg", vis2)
    print(f"Stage 2 saved: 最小外接矩形")

    # 阶段 3: 每个矩形角点对应的轮廓最近点 + 距离
    vis3 = image.copy()
    cv2.drawContours(vis3, [contour], -1, (0, 255, 0), 2)
    cv2.drawContours(vis3, [box], 0, (255, 0, 0), 2)

    closest_points = []
    min_distances = []

    for i, corner in enumerate(box):
        distances = np.linalg.norm(contour_points - corner, axis=1)
        min_dist = np.min(distances)
        min_idx = np.argmin(distances)
        closest_pt = tuple(contour_points[min_idx])

        closest_points.append(closest_pt)
        min_distances.append(min_dist)

        # 绘制矩形角点（黄色）
        cv2.circle(vis3, tuple(corner), 6, (0, 255, 255), -1)
        # 绘制轮廓最近点（橙色）
        cv2.circle(vis3, closest_pt, 8, (0, 165, 255), -1)
        # 连线
        cv2.line(vis3, tuple(corner), closest_pt, (200, 200, 200), 1)
        # 标注距离
        mid = ((corner[0]+closest_pt[0])//2, (corner[1]+closest_pt[1])//2)
        cv2.putText(vis3, f"{min_dist:.0f}px", mid,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(vis3, f"Stage 3: Closest Contour Points to Each Corner", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.imwrite(f"{output_dir}/contour_{contour_idx}_stage3_closest_points.jpg", vis3)
    print(f"Stage 3 saved: 各角点最近轮廓点，距离: {[f'{d:.1f}' for d in min_distances]}")

    # 阶段 4: 找出空缺角（距离最大），选出 3 个有效角点
    vis4 = image.copy()
    cv2.drawContours(vis4, [contour], -1, (0, 255, 0), 2)
    cv2.drawContours(vis4, [box], 0, (255, 0, 0), 2)

    max_dist_idx = np.argmax(min_distances)

    for i, (corner, closest_pt, dist) in enumerate(zip(box, closest_points, min_distances)):
        if i == max_dist_idx:
            # 空缺角：红色 X
            cv2.circle(vis4, tuple(corner), 10, (0, 0, 255), 2)
            cv2.line(vis4, (corner[0]-8, corner[1]-8), (corner[0]+8, corner[1]+8), (0, 0, 255), 2)
            cv2.line(vis4, (corner[0]+8, corner[1]-8), (corner[0]-8, corner[1]+8), (0, 0, 255), 2)
            cv2.putText(vis4, f"SKIP({dist:.0f}px)", (corner[0]+10, corner[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # 有效角：绿色圆圈 + 对应轮廓点
            cv2.circle(vis4, tuple(corner), 8, (0, 255, 255), -1)
            cv2.circle(vis4, closest_pt, 10, (0, 255, 0), -1)
            cv2.circle(vis4, closest_pt, 10, (255, 255, 255), 2)
            cv2.line(vis4, tuple(corner), closest_pt, (200, 200, 200), 1)

    cv2.putText(vis4, f"Stage 4: Skip Max-Dist Corner (red), Keep 3 (green)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(f"{output_dir}/contour_{contour_idx}_stage4_skip_corner.jpg", vis4)
    print(f"Stage 4 saved: 空缺角为 C{max_dist_idx}（距离 {min_distances[max_dist_idx]:.1f}px）")

    # 阶段 5: 最终 3 个关键点
    result = find_l_shape_keypoints(contour)

    if result is not None:
        vis5 = image.copy()
        cv2.drawContours(vis5, [contour], -1, (0, 200, 0), 1)
        cv2.drawContours(vis5, [box], 0, (255, 0, 0), 2)

        keypoints = result['keypoints']
        labels = ['端点1', '拐点', '端点2']
        colors = [(255, 128, 0), (0, 0, 255), (255, 0, 128)]

        for kp, label, color in zip(keypoints, labels, colors):
            cv2.circle(vis5, kp, 12, color, -1)
            cv2.circle(vis5, kp, 12, (255, 255, 255), 2)
            cv2.putText(vis5, label, (kp[0]+15, kp[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for line in result['lines']:
            x1, y1, x2, y2 = line
            cv2.line(vis5, (x1, y1), (x2, y2), (255, 255, 0), 2)

        cv2.putText(vis5, f"Stage 5: Final 3 Keypoints", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(f"{output_dir}/contour_{contour_idx}_stage5_final.jpg", vis5)
        print(f"Stage 5 saved: 最终关键点")
        for label, kp in zip(labels, keypoints):
            print(f"  {label}: {kp}")
    else:
        print(f"Stage 5: 未能提取关键点")

    print(f"\n所有阶段图片已保存到 {output_dir}/")


def visualize_l_shape_keypoints(image_path: str, output_path: str = None):
    """
    可视化 L 型关键点检测效果（改进版：基于凸包极值点）

    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径，默认保存到 output/ 目录
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return

    # 提取红色区域
    mask = extract_red_mask(image)

    # 检测轮廓
    contours = find_contours(mask)
    contours = filter_contours_by_area(contours, min_area=200, max_area=10000)

    print(f"检测到 {len(contours)} 个轮廓")

    # 创建可视化图像
    vis = image.copy()

    detected_count = 0

    for idx, contour in enumerate(contours):
        # 提取 L 型关键点
        result = find_l_shape_keypoints(contour)

        if result is None:
            # 未检测到 L 型，绘制灰色轮廓
            cv2.drawContours(vis, [contour], -1, (128, 128, 128), 1)
            continue

        detected_count += 1

        # 绘制原始轮廓（浅绿色，细线）
        cv2.drawContours(vis, [contour], -1, (0, 200, 0), 1)

        # 绘制凸包（黄色虚线）
        hull = cv2.convexHull(contour)
        cv2.drawContours(vis, [hull], -1, (0, 255, 255), 2)

        # 绘制拟合的两条直线（蓝色粗线）
        for line in result['lines']:
            x1, y1, x2, y2 = line
            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # 绘制 3 个关键点（红色大圆圈 + 标签）
        keypoints = result['keypoints']
        labels = ['端点1', '拐点', '端点2']

        for i, (kp, label) in enumerate(zip(keypoints, labels)):
            # 红色实心圆
            cv2.circle(vis, kp, 10, (0, 0, 255), -1)
            # 白色边框
            cv2.circle(vis, kp, 10, (255, 255, 255), 2)
            # 标签
            cv2.putText(vis, label, (kp[0]+15, kp[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 标注编号（在轮廓中心）
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(vis, f"L#{detected_count}", (cx-20, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # 添加统计信息
    info = f"检测到 {detected_count}/{len(contours)} 个 L 型灯条"
    cv2.putText(vis, info, (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # 保存结果
    if output_path is None:
        os.makedirs('output', exist_ok=True)
        basename = os.path.basename(image_path)
        output_path = f"output/{basename.split('.')[0]}_keypoints_v2.jpg"

    cv2.imwrite(output_path, vis)
    print(f"\n=== 检测结果 ===")
    print(f"检测到 {detected_count} 个 L 型灯条")
    print(f"可视化结果已保存到: {output_path}")

    # 输出每个 L 型的关键点坐标
    for idx, contour in enumerate(contours):
        result = find_l_shape_keypoints(contour)
        if result is not None:
            print(f"\nL 型 #{idx+1} 关键点:")
            for i, kp in enumerate(result['keypoints']):
                print(f"  {['端点1', '拐点', '端点2'][i]}: {kp}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 test_corner_detection.py <image_path> [contour_idx]")
        print("  contour_idx: 可选，指定要详细分析的轮廓编号（从1开始）")
        sys.exit(1)

    image_path = sys.argv[1]

    # 如果路径不是绝对路径，添加 data/ 前缀
    if not os.path.isabs(image_path) and not os.path.exists(image_path):
        image_path = os.path.join('data', image_path)

    # 如果指定了轮廓编号，进行分阶段可视化
    if len(sys.argv) >= 3:
        target_contour_idx = int(sys.argv[2])

        # 读取图像并提取轮廓
        image = cv2.imread(image_path)
        mask = extract_red_mask(image)
        contours = find_contours(mask)
        contours = filter_contours_by_area(contours, min_area=200, max_area=10000)

        print(f"检测到 {len(contours)} 个轮廓")

        if target_contour_idx < 1 or target_contour_idx > len(contours):
            print(f"错误: 轮廓编号必须在 1-{len(contours)} 之间")
            sys.exit(1)

        # 对指定轮廓进行分阶段可视化
        contour = contours[target_contour_idx - 1]
        visualize_stages_for_contour(image, contour, target_contour_idx)
    else:
        # 正常的整体可视化
        visualize_l_shape_keypoints(image_path)
