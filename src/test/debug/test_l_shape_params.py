#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试不同参数对 L 型轮廓的直线段提取效果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from perception import extract_red_mask, find_contours, filter_contours_by_area
from test.debug.debug_line_extraction import extract_line_segments, visualize_line_segments
import cv2

image_path = sys.argv[1] if len(sys.argv) > 1 else 'data/1.png'
image = cv2.imread(image_path)
mask = extract_red_mask(image)
contours = find_contours(mask)
candidates = filter_contours_by_area(contours, min_area=100, max_area=50000)

output_dir = "output/debug_line_segments/parameter_test"
os.makedirs(output_dir, exist_ok=True)

print("测试不同参数对 L 型轮廓的影响\n")

# 测试 Candidate #2 和 #3（L 型轮廓）
for idx in [1, 2]:  # 索引 1, 2 对应 Candidate #2, #3
    contour = candidates[idx]
    candidate_idx = idx + 1

    print(f"{'='*60}")
    print(f"Candidate #{candidate_idx} (点数: {len(contour)})")
    print(f"{'='*60}\n")

    # 测试不同参数组合
    test_params = [
        (3.0, 10, "default"),
        (5.0, 8, "relaxed"),
        (7.0, 5, "very_relaxed"),
    ]

    for grad_th, min_len, label in test_params:
        angles, gradient, lines, straight_thr = extract_line_segments(
            contour,
            gradient_threshold=grad_th,
            min_length=min_len
        )

        print(f"参数: gradient_threshold={grad_th}, min_length={min_len}, effective |grad|≤{straight_thr:.4f}")
        if lines:
            print(f"  ✓ 检测到 {len(lines)} 条线段")
            for line_idx, line in enumerate(lines):
                print(f"    Line {line_idx+1}: {line.length} 点, 角度={line.avg_angle:.1f}°")

            # 可视化
            vis_path = visualize_line_segments(
                image.copy(), contour, f"{candidate_idx}_{label}",
                angles, gradient, lines, output_dir,
                gradient_threshold=straight_thr,
                gradient_fallback_ref=grad_th,
            )
            print(f"  可视化: {os.path.basename(vis_path)}")
        else:
            print(f"  ✗ 未检测到线段")
        print()

print(f"\n所有输出保存到: {output_dir}")
