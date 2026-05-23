#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 candidate_*_warped.jpg 批量绘制：凸包缺陷角点 + 外 L 三外角点（与 keypoints 脚本一致）。"""

import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from calculation.contour_corner_detector import extract_l_shape_corners  # noqa: E402
from calculation.corner_detector import find_l_shape_keypoints, find_outer_corners  # noqa: E402
from perception import (  # noqa: E402
    extract_red_mask_from_gray_contours,
    find_contours,
    filter_contours_by_area,
)


def process_warped_image(warped_path: str, out_path: str) -> None:
    img = cv2.imread(warped_path)
    if img is None:
        print(f"skip read fail: {warped_path}")
        return

    h, w = img.shape[:2]
    image_size = max(h, w)

    mask = extract_red_mask_from_gray_contours(img)
    contours = find_contours(mask)
    contours = filter_contours_by_area(
        contours, min_area=50, max_area=h * w)
    vis = img.copy()

    if not contours:
        cv2.putText(vis, "no contour", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(out_path, vis)
        print(f"{os.path.basename(warped_path)}: no contour")
        return

    proc = max(contours, key=cv2.contourArea).astype(np.int32)
    cv2.drawContours(vis, [proc], -1, (0, 180, 0), 1)

    kp = find_l_shape_keypoints(proc)
    outer_source = "min_rect"
    if kp is None:
        lres = extract_l_shape_corners(proc, img, image_size=image_size)
        if lres is not None:
            kp = {
                "keypoints": lres["outer_keypoints"],
                "lines": [],
                "min_rect": None,
            }
            outer_source = lres.get("method", "extract_l_shape")

    if kp is not None:
        box = kp.get("min_rect")
        if box is not None:
            cv2.drawContours(vis, [box], -1, (255, 0, 0), 1)
        for line in kp.get("lines") or []:
            x1, y1, x2, y2 = line
            cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        labels = ["E1", "K", "E2"]
        colors = [(255, 128, 0), (0, 0, 255), (255, 0, 128)]
        for p, lab, col in zip(kp["keypoints"], labels, colors):
            cv2.circle(vis, (int(p[0]), int(p[1])), 8, col, -1)
            cv2.putText(vis, lab, (int(p[0]) + 8, int(p[1]) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(vis, f"outer: {outer_source}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(vis, "outer: FAIL", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    hull_pts = find_outer_corners(proc, defect_threshold=0.05)
    for p in hull_pts:
        cv2.circle(vis, (int(p[0]), int(p[1])), 4, (255, 255, 255), -1)
        cv2.circle(vis, (int(p[0]), int(p[1])), 5, (0, 0, 0), 1)
    cv2.putText(
        vis,
        f"hull defect: {len(hull_pts)}",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (220, 220, 220), 1,
    )

    cv2.imwrite(out_path, vis)
    print(
        f"{os.path.basename(warped_path)} -> {os.path.basename(out_path)} "
        f"(outer={'ok' if kp else 'fail'}, hull={len(hull_pts)})"
    )


def main():
    if len(sys.argv) < 2:
        print(
            "用法: python3 debug_warped_all_corners.py <目录>   "
            "处理目录下 candidate_*_warped.jpg"
        )
        sys.exit(1)
    folder = sys.argv[1]
    paths = sorted(glob.glob(os.path.join(folder, "candidate_*_warped.jpg")))
    if not paths:
        print(f"未找到 candidate_*_warped.jpg: {folder}")
        sys.exit(1)
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        out_p = os.path.join(folder, f"{base}_all_corners.jpg")
        process_warped_image(p, out_p)


if __name__ == "__main__":
    main()
