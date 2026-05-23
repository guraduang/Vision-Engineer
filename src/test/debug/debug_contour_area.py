#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试脚本 - 查看轮廓面积
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from perception.red_extractor import extract_red_mask
from perception.contour_detector import find_contours

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 debug_contour_area.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isabs(image_path) and not os.path.exists(image_path):
        image_path = os.path.join('data', image_path)

    image = cv2.imread(image_path)
    mask = extract_red_mask(image)
    contours = find_contours(mask)

    print(f"\n检测到 {len(contours)} 个轮廓")
    print("\n轮廓面积列表：")

    areas = []
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        areas.append((idx+1, area))
        print(f"  轮廓 #{idx+1}: {area:.0f} px²")

    areas.sort(key=lambda x: x[1], reverse=True)
    print(f"\n按面积排序（从大到小）：")
    for idx, area in areas:
        print(f"  轮廓 #{idx}: {area:.0f} px²")

    print(f"\n当前筛选参数: min_area=500, max_area=50000")
    filtered = [a for _, a in areas if 500 <= a <= 50000]
    print(f"筛选后剩余: {len(filtered)} 个轮廓")
