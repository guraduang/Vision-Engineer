#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
L 型匹配调试工具
展示每个候选轮廓的链码匹配过程、角度梯度分析
输出组织：output/debug_l_matching/<image_name>/
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
from calculation import is_l_shape_chaincode
from calculation.contour_corner_detector import (
    compute_contour_angles,
    compute_angle_gradient,
    detect_monotonic_segments
)


def visualize_candidate_analysis(image, contour, candidate_idx, epsilon_factor,
                                   is_l, info, output_dir):
    """可视化单个候选轮廓的详细分析"""

    if not info:
        return

    # 创建子图
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 原图 + 轮廓
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.drawContours(image.copy(), [contour], -1, (0, 255, 0), 2)
    approx = info.get('approx', contour)
    if approx is not None:
        pts = approx.reshape(-1, 2)
        ax1.plot(pts[:, 0], pts[:, 1], 'ro-', markersize=8, linewidth=2)
        for i, (x, y) in enumerate(pts):
            ax1.annotate(str(i+1), (x, y), color='yellow', fontsize=10,
                        fontweight='bold', xytext=(5, 5), textcoords='offset points')
    ax1.set_title(f'Candidate #{candidate_idx} (epsilon={epsilon_factor})', fontsize=11)
    ax1.axis('off')

    # 2. 链码可视化
    ax2 = fig.add_subplot(gs[0, 1])
    chain = info.get('chain', [])
    simplified = info.get('simplified', [])
    normalized = info.get('normalized', [])

    text_info = f"Chain Code Analysis\n"
    text_info += f"{'='*40}\n"
    text_info += f"Raw chain: {chain}\n"
    text_info += f"Simplified: {simplified}\n"
    text_info += f"Normalized: {normalized}\n"
    text_info += f"\nMatch ratio: {info.get('match_ratio', 0):.3f}\n"
    text_info += f"Convex ratio: {info.get('convex_ratio', 0):.3f}\n"
    text_info += f"Pair matches: {info.get('pair_matches', 0)}\n"
    text_info += f"\nIs L-shape: {is_l}"

    ax2.text(0.05, 0.95, text_info, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.axis('off')

    # 3. 边长分析
    ax3 = fig.add_subplot(gs[0, 2])
    lengths = info.get('lengths', [])
    matched_pairs = info.get('matched_pairs', [])

    text_lengths = "Edge Lengths\n"
    text_lengths += f"{'='*40}\n"
    for i, length in enumerate(lengths):
        text_lengths += f"Edge {i+1}: {length:.1f} px\n"

    text_lengths += f"\nMatched Pairs ({len(matched_pairs)}):\n"
    for i, j, l1, l2 in matched_pairs:
        ratio = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 0
        text_lengths += f"  {i+1}-{j+1}: {l1:.1f} vs {l2:.1f} (ratio={ratio:.2f})\n"

    ax3.text(0.05, 0.95, text_lengths, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax3.axis('off')

    # 4-6. 角度梯度分析（如果轮廓足够大）
    if len(contour) > 10:
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            angles = compute_contour_angles(contour, (cx, cy))
            gradient = compute_angle_gradient(angles)
            segments = detect_monotonic_segments(gradient, threshold=5.0, min_length=5)

            angles_flat = angles.flatten()
            gradient_flat = gradient.flatten()
            indices = np.arange(len(angles_flat))

            # 4. 轮廓点可视化
            ax4 = fig.add_subplot(gs[1, :])
            contour_pts = contour.reshape(-1, 2)
            ax4.plot(contour_pts[:, 0], contour_pts[:, 1], 'b-', linewidth=1, alpha=0.7)
            ax4.scatter(contour_pts[:, 0], contour_pts[:, 1], c=gradient_flat,
                       cmap='RdYlGn_r', s=20, vmin=-20, vmax=20)
            ax4.plot(cx, cy, 'r*', markersize=15)
            ax4.set_title(f'Contour Points (colored by gradient)', fontsize=11)
            ax4.axis('equal')
            ax4.grid(True, alpha=0.3)

            # 5. 角度曲线
            ax5 = fig.add_subplot(gs[2, 0:2])
            ax5.plot(indices, angles_flat, 'b-', linewidth=1.5, label='Angle')
            ax5.set_xlabel('Contour point index')
            ax5.set_ylabel('Angle (degrees)')
            ax5.set_title(f'Angle Curve (range: [{angles_flat.min():.1f}, {angles_flat.max():.1f}])')
            ax5.grid(True, alpha=0.3)
            ax5.legend()

            # 6. 梯度曲线 + 单调区间
            ax6 = fig.add_subplot(gs[2, 2])
            ax6.plot(indices, gradient_flat, 'g-', linewidth=1.5, label='Gradient')
            ax6.axhline(5.0, color='orange', linestyle='--', linewidth=1, label='Threshold=5.0')
            ax6.axhline(-5.0, color='orange', linestyle='--', linewidth=1)

            # 标注单调区间
            for seg in segments:
                ax6.axvspan(seg[0], seg[1], alpha=0.2, color='cyan')

            ax6.set_xlabel('Contour point index')
            ax6.set_ylabel('Gradient')
            ax6.set_title(f'Gradient: {len(segments)} monotonic segments')
            ax6.grid(True, alpha=0.3)
            ax6.legend()

    # 保存
    result_color = 'green' if is_l else 'red'
    plt.suptitle(f'Candidate #{candidate_idx} Analysis - Result: {is_l}',
                fontsize=14, fontweight='bold', color=result_color)

    output_path = os.path.join(output_dir, f'candidate_{candidate_idx:02d}_eps{epsilon_factor}.jpg')
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return output_path


def debug_l_shape_matching(image_path: str):
    """调试 L 型匹配流程"""

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return

    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = f"output/debug_l_matching/{basename}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Debugging L-shape Matching: {basename} ===")
    print(f"Output directory: {output_dir}\n")

    # 提取红色mask
    mask = extract_red_mask(image)
    cv2.imwrite(os.path.join(output_dir, "01_red_mask.jpg"), mask)

    # 查找轮廓
    contours = find_contours(mask)
    print(f"Total contours: {len(contours)}")

    # 筛选候选轮廓
    candidates = filter_contours_by_area(contours, min_area=100, max_area=50000)
    print(f"Candidate contours: {len(candidates)}\n")

    # 绘制所有候选轮廓
    all_candidates_img = image.copy()
    for idx, contour in enumerate(candidates):
        cv2.drawContours(all_candidates_img, [contour], -1, (0, 255, 255), 2)
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(all_candidates_img, f"#{idx+1}", (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "02_all_candidates.jpg"), all_candidates_img)

    # 逐个分析候选轮廓
    l_shapes = []
    candidate_count = 0

    for contour in candidates:
        candidate_count += 1
        print(f"\n{'='*60}")
        print(f"Analyzing Candidate #{candidate_count}")
        print(f"{'='*60}")

        for epsilon_factor in [0.02, 0.015, 0.025, 0.03]:
            is_l, approx, info = is_l_shape_chaincode(contour, image, epsilon_factor)

            if info:
                print(f"\nEpsilon={epsilon_factor}:")
                print(f"  Points: {len(info.get('approx', []))}")
                print(f"  Convex ratio: {info.get('convex_ratio', 0):.3f}")
                print(f"  Match ratio: {info.get('match_ratio', 0):.3f}")
                print(f"  Pair matches: {info.get('pair_matches', 0)}")
                print(f"  Is L-shape: {is_l}")

                # 可视化分析
                vis_path = visualize_candidate_analysis(
                    image.copy(), contour, candidate_count, epsilon_factor,
                    is_l, info, output_dir
                )
                print(f"  Visualization saved: {os.path.basename(vis_path)}")

            if is_l:
                l_shapes.append({
                    'contour': contour,
                    'approx': approx,
                    'info': info,
                    'candidate_idx': candidate_count
                })
                print(f"  ✓ Accepted as L-shape")
                break

    # 绘制最终结果
    result_img = image.copy()
    for idx, shape in enumerate(l_shapes):
        contour = shape['contour']
        approx = shape['approx']

        cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(result_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(result_img, f"L#{idx+1}", (cx+10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(output_dir, "99_final_result.jpg"), result_img)

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total candidates analyzed: {candidate_count}")
    print(f"L-shapes detected: {len(l_shapes)}")
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  - 01_red_mask.jpg: Red extraction result")
    print(f"  - 02_all_candidates.jpg: All candidate contours")
    print(f"  - candidate_XX_epsY.YY.jpg: Detailed analysis for each candidate")
    print(f"  - 99_final_result.jpg: Final detection result")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 debug_l_shape_matching.py <image_path>")
        sys.exit(1)

    debug_l_shape_matching(sys.argv[1])
