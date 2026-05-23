#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试脚本 - 可视化所有轮廓（包括被筛选掉的）
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from perception.red_extractor import extract_red_mask
from perception.contour_detector import find_contours

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 debug_all_contours.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isabs(image_path) and not os.path.exists(image_path):
        image_path = os.path.join('data', image_path)

    image = cv2.imread(image_path)
    mask = extract_red_mask(image)
    contours = find_contours(mask)

    print(f"\n检测到 {len(contours)} 个轮廓")

    # 创建可视化图像
    vis = image.copy()

    # 按面积分类
    large_contours = []  # > 50000
    valid_contours = []  # 500-50000
    small_contours = []  # < 500

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area > 50000:
            large_contours.append((idx+1, contour, area))
        elif area >= 500:
            valid_contours.append((idx+1, contour, area))
        else:
            small_contours.append((idx+1, contour, area))

    # 绘制不同类别的轮廓
    # 大轮廓（紫色）
    for idx, contour, area in large_contours:
        cv2.drawContours(vis, [contour], -1, (255, 0, 255), 2)
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(vis, f"#{idx}({area:.0f})", (cx-30, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # 有效轮廓（绿色）
    for idx, contour, area in valid_contours:
        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(vis, f"#{idx}({area:.0f})", (cx-30, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 小轮廓（红色）
    for idx, contour, area in small_contours:
        cv2.drawContours(vis, [contour], -1, (0, 0, 255), 1)
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(vis, f"#{idx}({area:.0f})", (cx-20, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 添加图例
    cv2.putText(vis, f"Green: Valid (500-50000) = {len(valid_contours)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, f"Red: Too Small (<500) = {len(small_contours)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis, f"Purple: Too Large (>50000) = {len(large_contours)}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # 保存结果
    os.makedirs('output', exist_ok=True)
    basename = os.path.basename(image_path)
    output_path = f"output/{basename.split('.')[0]}_all_contours.jpg"
    cv2.imwrite(output_path, vis)

    print(f"\n分类统计：")
    print(f"  绿色（有效）: {len(valid_contours)} 个")
    print(f"  红色（太小）: {len(small_contours)} 个")
    print(f"  紫色（太大）: {len(large_contours)} 个")
    print(f"\n可视化结果已保存到: {output_path}")
