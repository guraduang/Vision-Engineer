#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化不同配置对角点检测的影响
生成对比图像，展示不同配置下的检测结果
"""

import cv2
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.calculation.contour_corner_detector import extract_l_shape_corners
from src.perception.red_extractor import extract_red_mask
from src.utils.config_loader import get_config


def warp_l_shape_to_square(image: np.ndarray, contour: np.ndarray, square_size: int = 300):
    """将 L 型通过最小外接矩形透视变换成正方形"""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.float32(box)

    dst_pts = np.float32([
        [0, 0],
        [square_size, 0],
        [square_size, square_size],
        [0, square_size]
    ])

    M = cv2.getPerspectiveTransform(box, dst_pts)
    warped = cv2.warpPerspective(image, M, (square_size, square_size))

    return warped, M


def visualize_config_comparison(image_path: str, output_dir: str = "output/config_comparison"):
    """
    可视化不同配置的角点检测效果对比

    Args:
        image_path: 测试图像路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    image_name = os.path.basename(image_path).split('.')[0]

    # 提取红色掩码
    mask = extract_red_mask(image)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("未找到轮廓")
        return

    # 按面积排序，选择最大的轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 配置文件列表
    profiles = ['default', 'high_precision', 'fast', 'experimental']
    config = get_config()

    # 处理前3个轮廓
    for contour_idx, contour in enumerate(contours[:3]):
        # 透视变换
        warped, M = warp_l_shape_to_square(image, contour)
        if warped is None:
            continue

        # 在变换后的图像中重新提取轮廓
        warped_mask = extract_red_mask(warped)
        warped_contours, _ = cv2.findContours(warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(warped_contours) == 0:
            continue

        warped_contour = max(warped_contours, key=cv2.contourArea)

        # 为每个配置生成可视化
        vis_images = []
        titles = []

        for profile in profiles:
            # 切换配置
            config.set_profile(profile)

            # 提取角点
            result = extract_l_shape_corners(warped_contour, warped)

            # 创建可视化图像
            vis = warped.copy()

            if result is not None:
                # 绘制轮廓
                cv2.drawContours(vis, [warped_contour], -1, (0, 255, 0), 2)

                # 绘制外轮廓角点
                if 'outer_keypoints' in result:
                    for i, pt in enumerate(result['outer_keypoints']):
                        cv2.circle(vis, tuple(map(int, pt)), 5, (0, 0, 255), -1)
                        cv2.putText(vis, f"O{i+1}", (int(pt[0])+10, int(pt[1])),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 绘制内拐点
                if 'inner_corner' in result and result['inner_corner'] is not None:
                    pt = result['inner_corner']
                    cv2.circle(vis, tuple(map(int, pt)), 5, (255, 0, 0), -1)
                    cv2.putText(vis, "Inner", (int(pt[0])+10, int(pt[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # 标注使用的方法
                method = result.get('method', 'unknown')
                method_color = (0, 255, 0) if method == 'new' else (0, 165, 255)
                method_text = "NEW" if method == 'new' else "FALLBACK"
                cv2.putText(vis, method_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, method_color, 2)

            # 添加配置名称
            cv2.putText(vis, profile.upper(), (10, 290),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            vis_images.append(vis)
            titles.append(profile)

        # 拼接成 2x2 网格
        if len(vis_images) == 4:
            row1 = np.hstack([vis_images[0], vis_images[1]])
            row2 = np.hstack([vis_images[2], vis_images[3]])
            grid = np.vstack([row1, row2])

            # 添加标题
            title_height = 40
            title_img = np.zeros((title_height, grid.shape[1], 3), dtype=np.uint8)
            title_text = f"{image_name} - Contour #{contour_idx+1}"
            cv2.putText(title_img, title_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            final_img = np.vstack([title_img, grid])

            # 保存
            output_path = os.path.join(output_dir, f"{image_name}_contour_{contour_idx+1}_comparison.jpg")
            cv2.imwrite(output_path, final_img)
            print(f"已保存: {output_path}")


def visualize_all_images():
    """可视化所有测试图像"""
    test_images = [
        '/home/workspace/vision/data/1.png',
        '/home/workspace/vision/data/2.png',
        '/home/workspace/vision/data/3.png'
    ]

    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n处理图像: {os.path.basename(image_path)}")
            visualize_config_comparison(image_path)
        else:
            print(f"图像不存在: {image_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 可视化单个图像
        visualize_config_comparison(sys.argv[1])
    else:
        # 可视化所有图像
        visualize_all_images()
