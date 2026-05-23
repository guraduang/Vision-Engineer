#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
L 型灯条检测与透视变换测试
功能：
1. 检测 L 型灯条并可视化（标注轮廓、中心点、编号）
2. 对每个检测到的 L 型灯条进行透视变换
3. 展示透视变换前后的对比效果
"""

import cv2
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from perception import (
    extract_red_mask,
    find_contours,
    filter_contours_by_area
)
from calculation import is_l_shape_chaincode, warp_to_frontal


def visualize_detection(image: np.ndarray, l_shapes: list, output_path: str):
    """
    可视化 L 型灯条检测结果

    Args:
        image: 原图像
        l_shapes: 检测到的 L 型灯条列表
        output_path: 输出路径
    """
    result = image.copy()

    for idx, shape in enumerate(l_shapes):
        contour = shape['contour']
        approx = shape['approx']

        # 绘制轮廓（绿色）
        cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)

        # 绘制中心点（红色）
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(result, f"#{idx+1}", (cx+10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 绘制角点（蓝色）
        for point in approx:
            pt = tuple(point[0])
            cv2.circle(result, pt, 3, (255, 0, 0), -1)

    # 添加统计信息
    info_text = f"Detected: {len(l_shapes)} L-shapes"
    cv2.putText(result, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(output_path, result)
    print(f"检测可视化已保存到: {output_path}")


def visualize_perspective_transform(image: np.ndarray, l_shapes: list, output_dir: str):
    """
    可视化透视变换效果

    Args:
        image: 原图像
        l_shapes: 检测到的 L 型灯条列表
        output_dir: 输出目录
    """
    for idx, shape in enumerate(l_shapes):
        contour = shape['contour']

        # 执行透视变换
        warped, warped_contour = warp_to_frontal(image, contour)

        if warped is None:
            print(f"L-shape #{idx+1}: 透视变换失败")
            continue

        # 在原图上标注该轮廓
        original_marked = image.copy()
        cv2.drawContours(original_marked, [contour], -1, (0, 255, 0), 2)

        # 获取最小外接矩形用于标注
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(original_marked, [box], -1, (255, 0, 0), 2)

        # 在原图上标注编号
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(original_marked, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(original_marked, f"#{idx+1}", (cx+10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 调整大小以便对比显示
        h_orig, w_orig = original_marked.shape[:2]
        h_warp, w_warp = warped.shape[:2]

        # 统一高度
        target_height = 300
        scale_orig = target_height / h_orig
        scale_warp = target_height / h_warp

        resized_orig = cv2.resize(original_marked,
                                  (int(w_orig * scale_orig), target_height))
        resized_warp = cv2.resize(warped,
                                  (int(w_warp * scale_warp), target_height))

        # 添加标签
        label_orig = np.zeros((40, resized_orig.shape[1], 3), dtype=np.uint8)
        cv2.putText(label_orig, f"Original #{idx+1}", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        label_warp = np.zeros((40, resized_warp.shape[1], 3), dtype=np.uint8)
        cv2.putText(label_warp, f"Warped #{idx+1}", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 拼接标签和图像
        img_with_label_orig = np.vstack([label_orig, resized_orig])
        img_with_label_warp = np.vstack([label_warp, resized_warp])

        # 水平拼接对比图
        comparison = np.hstack([img_with_label_orig, img_with_label_warp])

        # 保存对比图
        output_path = os.path.join(output_dir, f"perspective_{idx+1}.jpg")
        cv2.imwrite(output_path, comparison)
        print(f"L-shape #{idx+1}: 透视变换对比图已保存到 {output_path}")

        # 单独保存变换后的图像
        warped_path = os.path.join(output_dir, f"warped_{idx+1}.jpg")
        cv2.imwrite(warped_path, warped)
        print(f"L-shape #{idx+1}: 变换后图像已保存到 {warped_path}")


def process_image(image_path: str):
    """
    处理单张图像：检测 L 型灯条并进行透视变换

    Args:
        image_path: 图像路径
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(image_path)}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"{'='*60}\n")

    # 提取红色mask
    mask = extract_red_mask(image)

    # 查找轮廓
    contours = find_contours(mask)
    print(f"Total contours: {len(contours)}")

    # 筛选候选轮廓
    candidates = filter_contours_by_area(contours, min_area=100, max_area=50000)
    print(f"Candidate contours: {len(candidates)}")

    # 检测L型
    l_shapes = []
    candidate_count = 0
    for contour in candidates:
        candidate_count += 1
        for epsilon_factor in [0.02, 0.015, 0.025, 0.03]:
            is_l, approx, info = is_l_shape_chaincode(contour, image, epsilon_factor)

            if is_l:
                l_shapes.append({
                    'contour': contour,
                    'approx': approx,
                    'info': info
                })
                print(f"✓ Candidate #{candidate_count}: L-shape detected (epsilon={epsilon_factor})")
                break

    # 去重（内外轮廓）
    unique_l_shapes = []
    used = set()
    for i, shape1 in enumerate(l_shapes):
        if i in used:
            continue

        M1 = cv2.moments(shape1['contour'])
        if M1['m00'] == 0:
            continue
        cx1 = int(M1['m10'] / M1['m00'])
        cy1 = int(M1['m01'] / M1['m00'])

        is_duplicate = False
        for j, shape2 in enumerate(l_shapes):
            if i == j or j in used:
                continue

            M2 = cv2.moments(shape2['contour'])
            if M2['m00'] == 0:
                continue
            cx2 = int(M2['m10'] / M2['m00'])
            cy2 = int(M2['m01'] / M2['m00'])

            dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
            if dist < 20:
                used.add(j)
                is_duplicate = True

        if not is_duplicate:
            unique_l_shapes.append(shape1)

    print(f"\n{'='*60}")
    print(f"Detection Results: {len(unique_l_shapes)} L-shapes detected")
    print(f"{'='*60}\n")

    # 打印详细信息
    for idx, shape in enumerate(unique_l_shapes):
        info = shape['info']
        print(f"L-shape #{idx+1}:")
        print(f"  Chain code: {info['chain']}")
        print(f"  Simplified: {info['simplified']}")
        print(f"  Normalized: {info['normalized']}")
        print(f"  Match ratio: {info['match_ratio']:.2f}")
        print(f"  Convex ratio: {info['convex_ratio']:.3f}")
        print(f"  Pair matches: {info['pair_matches']}")
        print()

    # 创建输出目录
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(image_path))[0]

    # 可视化检测结果
    detection_output = os.path.join(output_dir, f"{basename}_detection.jpg")
    visualize_detection(image, unique_l_shapes, detection_output)

    # 可视化透视变换
    print(f"\n{'='*60}")
    print("Perspective Transform")
    print(f"{'='*60}\n")
    visualize_perspective_transform(image, unique_l_shapes, output_dir)

    print(f"\n{'='*60}")
    print("Processing Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_l_shape_with_perspective.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    process_image(image_path)
