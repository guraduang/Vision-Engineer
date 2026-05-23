#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比测试 - 新旧角点检测算法
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
from calculation.contour_corner_detector import extract_l_shape_corners


def compare_algorithms(image_path: str):
    """
    对比新旧角点检测算法

    Args:
        image_path: 输入图像路径
    """
    print(f"\n{'='*80}")
    print(f"对比测试: {image_path}")
    print(f"{'='*80}\n")

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return

    # 1. 红色提取
    mask = extract_red_mask(image)

    # 2. 轮廓检测
    contours = find_contours(mask)
    contours = filter_contours_by_area(contours, min_area=200, max_area=10000)

    print(f"检测到 {len(contours)} 个候选轮廓\n")

    # 创建输出目录
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = f"output/{basename}_compare"
    os.makedirs(output_dir, exist_ok=True)

    # 统计
    old_success = 0
    new_success = 0
    both_success = 0
    total_errors = []

    for i, contour in enumerate(contours):
        print(f"轮廓 #{i+1}/{len(contours)}")

        # 旧算法
        old_result = find_l_shape_keypoints(contour)
        if old_result is None:
            print(f"  旧算法: 失败")
            print()
            continue

        old_success += 1
        old_keypoints = old_result['keypoints']

        # 透视变换
        src_pts = np.float32(old_keypoints)
        dst_pts = np.float32([[0, 0], [0, 300], [300, 300]])
        matrix = cv2.getAffineTransform(src_pts, dst_pts)
        warped = cv2.warpAffine(image, matrix, (300, 300))

        # 在变换后的图像上提取轮廓
        warped_mask = extract_red_mask(warped)
        warped_contours = find_contours(warped_mask)

        if len(warped_contours) == 0:
            print(f"  变换后无轮廓")
            print()
            continue

        # 选择最大轮廓
        warped_contour = max(warped_contours, key=cv2.contourArea)

        # 新算法
        new_result = extract_l_shape_corners(warped_contour, warped, image_size=300)

        if new_result is None:
            print(f"  旧算法: 成功")
            print(f"  新算法: 失败")
            print()
            continue

        new_success += 1
        both_success += 1

        # 提取结果
        new_outer_keypoints = new_result['outer_keypoints']
        new_inner_corner = new_result['inner_corner']

        # 旧算法的内拐点检测（使用 Shi-Tomasi）
        from test_perspective_transform import detect_corners_in_warped, find_inner_corners_from_outer
        all_corners = detect_corners_in_warped(warped)
        old_inner_corner = find_inner_corners_from_outer(
            [[0, 0], [0, 300], [300, 300]], all_corners, warped
        )

        # 计算外轮廓角点的误差
        # 旧算法的外轮廓角点是固定的 [0,0], [0,300], [300,300]
        old_outer = np.array([[0, 0], [0, 300], [300, 300]])
        new_outer = np.array(new_outer_keypoints)

        # 计算平均误差
        errors = np.linalg.norm(old_outer - new_outer, axis=1)
        avg_error = np.mean(errors)
        max_error = np.max(errors)

        total_errors.append(avg_error)

        print(f"  旧算法: 成功")
        print(f"  新算法: 成功")
        print(f"  外轮廓角点误差: 平均={avg_error:.2f}px, 最大={max_error:.2f}px")
        print(f"  内拐点: 旧={'找到' if old_inner_corner else '未找到'}, "
              f"新={'找到' if new_inner_corner else '未找到'}")

        # 可视化对比
        vis = np.hstack([warped.copy(), warped.copy()])

        # 左侧：旧算法结果
        cv2.putText(vis, "Old Algorithm", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for idx, pt in enumerate([[0, 0], [0, 300], [300, 300]]):
            cv2.circle(vis, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(vis, f"O{idx+1}", (pt[0]+10, pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if old_inner_corner:
            cv2.circle(vis, old_inner_corner, 5, (255, 0, 255), -1)

        # 右侧：新算法结果
        cv2.putText(vis, "New Algorithm", (310, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for idx, pt in enumerate(new_outer_keypoints):
            pt_shifted = (pt[0] + 300, pt[1])
            cv2.circle(vis, pt_shifted, 5, (0, 0, 255), -1)
            cv2.putText(vis, f"N{idx+1}", (pt_shifted[0]+10, pt_shifted[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if new_inner_corner:
            inner_shifted = (new_inner_corner[0] + 300, new_inner_corner[1])
            cv2.circle(vis, inner_shifted, 5, (255, 0, 255), -1)

        # 添加误差信息
        cv2.putText(vis, f"Avg Error: {avg_error:.2f}px", (10, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 保存对比图
        output_path = f"{output_dir}/L{i+1:02d}_compare.jpg"
        cv2.imwrite(output_path, vis)

        print()

    # 总结
    print(f"{'='*80}")
    print(f"测试总结:")
    print(f"  总轮廓数: {len(contours)}")
    print(f"  旧算法成功: {old_success}/{len(contours)} ({old_success*100/len(contours):.1f}%)")
    print(f"  新算法成功: {new_success}/{len(contours)} ({new_success*100/len(contours):.1f}%)")
    print(f"  两者都成功: {both_success}/{len(contours)} ({both_success*100/len(contours):.1f}%)")
    if total_errors:
        print(f"  平均角点误差: {np.mean(total_errors):.2f}px")
        print(f"  最大角点误差: {np.max(total_errors):.2f}px")
    print(f"  结果保存到: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 compare_corner_algorithms.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # 如果路径不是绝对路径，添加 data/ 前缀
    if not os.path.isabs(image_path) and not os.path.exists(image_path):
        image_path = os.path.join('data', image_path)

    compare_algorithms(image_path)
