#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对二值 warped mask 检测 6 个角点：最大外轮廓 + approxPolyDP 二分 epsilon。"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _largest_external_contour(binary: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def approx_poly_dp_n_vertices(
        contour: np.ndarray,
        n: int = 6,
        max_iter: int = 64) -> Optional[np.ndarray]:
    """
    二分搜索 epsilon，使 cv2.approxPolyDP 返回恰好 n 个顶点（闭合多边形）。

    epsilon 增大 → 顶点数单调不增（对同一轮廓通常可二分）。
    """
    peri = cv2.arcLength(contour, True)
    if peri <= 1e-6:
        return None

    lo, hi = 1e-9 * peri, 0.5 * peri
    best_approx = None
    best_diff = 10**9

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        approx = cv2.approxPolyDP(contour, mid, True)
        c = len(approx)
        diff = abs(c - n)
        if diff < best_diff:
            best_diff = diff
            best_approx = approx
        if c == n:
            return approx
        if c > n:
            lo = mid
        else:
            hi = mid

    return best_approx if best_diff == 0 else None


def detect_six_corners_from_mask_path(
        mask_path: str,
        morph_close_ksize: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Returns:
        corners (6, 1, 2) int32 或 None
        contour 用于绘制的轮廓
        binary 二值图
    """
    gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None, np.array([])

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if morph_close_ksize and morph_close_ksize >= 3:
        k = morph_close_ksize | 1  # odd
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contour = _largest_external_contour(binary)
    n = 6
    if contour is None or len(contour) < n * 2:
        return None, contour, binary

    approx = approx_poly_dp_n_vertices(contour, n=n)
    if approx is None or len(approx) != n:
        return None, contour, binary
    return approx.astype(np.int32), contour, binary


def main():
    parser = argparse.ArgumentParser(
        description="二值 mask 上 approxPolyDP 检测 6 角点并保存可视化")
    parser.add_argument("mask_path", help="例如 candidate_01_warped_mask.jpg")
    parser.add_argument(
        "-o", "--output", default="", help="输出路径，默认与 mask 同目录 *_six_corners.jpg")
    parser.add_argument(
        "--morph", type=int, default=0,
        help="可选：形态学闭运算核大小（奇数，>=3），默认 0 关闭")
    args = parser.parse_args()

    corners, contour, binary = detect_six_corners_from_mask_path(
        args.mask_path, morph_close_ksize=args.morph)

    base_dir = os.path.dirname(os.path.abspath(args.mask_path))
    base_name = os.path.splitext(os.path.basename(args.mask_path))[0]
    out_path = args.output or os.path.join(
        base_dir, f"{base_name}_six_corners.jpg")

    h, w = binary.shape[:2]
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (0, 120, 0), 1)

    if corners is None:
        cv2.putText(
            vis, "6-corner approx FAILED", (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.imwrite(out_path, vis)
        print(f"失败: 无法得到恰好 6 个顶点 -> {out_path}")
        sys.exit(1)

    pts = corners.reshape(-1, 2)
    # 按轮廓几何顺序已排好；画折线闭合
    poly = pts.astype(np.int32)
    cv2.polylines(vis, [poly], True, (255, 180, 0), 1)

    for i, (x, y) in enumerate(pts):
        cx, cy = int(x), int(y)
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(vis, (cx, cy), 7, (255, 255, 255), 1)
        cv2.putText(
            vis, str(i + 1), (cx + 8, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imwrite(out_path, vis)
    print(f"6 角点 (x, y) 轮廓顺序:")
    for i, (x, y) in enumerate(pts):
        print(f"  {i + 1}: ({int(x)}, {int(y)})")
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
