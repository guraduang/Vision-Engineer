#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本 - L 型透视变换
将检测到的 L 型通过 3 个角点透视变换成正方形
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


def detect_corners_in_warped(warped: np.ndarray, quality_level: float = 0.01, min_distance: int = 5, max_corners: int = 10):
    """
    在变换后的图像中检测角点

    Args:
        warped: 变换后的图像
        quality_level: 角点质量阈值
        min_distance: 角点之间的最小距离
        max_corners: 最大角点数量

    Returns:
        角点列表 [(x, y), ...]
    """
    # 转换为灰度图
    if len(warped.shape) == 3:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped

    # 使用 Shi-Tomasi 角点检测
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=3
    )

    if corners is None:
        return []

    # 转换为整数坐标
    corners = corners.reshape(-1, 2).astype(np.int32)
    return [tuple(pt) for pt in corners]


def find_inner_corners_from_outer(outer_keypoints: list, all_corners: list, warped_image: np.ndarray, image_size: int = 300):
    """
    根据外轮廓的 3 个角点推算内轮廓的内拐点

    Args:
        outer_keypoints: 外轮廓的 3 个角点 [端点1, 拐点, 端点2]
        all_corners: 检测到的所有角点
        warped_image: 变换后的图像（用于检查角点是否在红色区域）
        image_size: 图像尺寸

    Returns:
        内拐点坐标，如果找不到返回 None
    """
    if len(outer_keypoints) != 3 or len(all_corners) == 0:
        return None

    outer_endpoint1, outer_corner, outer_endpoint2 = outer_keypoints

    # 1. 提取红色掩码，确保角点在红色区域内
    from perception.red_extractor import extract_red_mask
    red_mask = extract_red_mask(warped_image)

    # 2. 计算外拐点到两个外端点的向量
    vec_to_endpoint1 = np.array(outer_endpoint1) - np.array(outer_corner)
    vec_to_endpoint2 = np.array(outer_endpoint2) - np.array(outer_corner)

    # 计算外轮廓两条边的平均长度
    outer_len1 = np.linalg.norm(vec_to_endpoint1)
    outer_len2 = np.linalg.norm(vec_to_endpoint2)
    avg_outer_len = (outer_len1 + outer_len2) / 2

    # 3. 计算 L 型的内侧区域中心点
    triangle_center = (np.array(outer_corner) + np.array(outer_endpoint1) + np.array(outer_endpoint2)) / 3

    # 4. 过滤候选点：
    #    - 不是外轮廓的 3 个点
    #    - 必须在红色区域内
    #    - 距离外拐点在合理范围内
    inner_candidates = []
    for corner in all_corners:
        # 排除外轮廓点
        is_outer = False
        for outer_pt in outer_keypoints:
            if np.linalg.norm(np.array(corner) - np.array(outer_pt)) < 30:
                is_outer = True
                break
        if is_outer:
            continue

        # 检查是否在红色区域内
        x, y = corner
        if x < 0 or x >= image_size or y < 0 or y >= image_size:
            continue
        if red_mask[y, x] == 0:  # 不在红色区域
            continue

        # 距离检查
        dist_to_outer_corner = np.linalg.norm(np.array(corner) - np.array(outer_corner))
        if dist_to_outer_corner < avg_outer_len * 0.25 or dist_to_outer_corner > avg_outer_len * 0.75:
            continue

        inner_candidates.append(corner)

    if len(inner_candidates) == 0:
        return None

    # 5. 从候选点中选择最佳内拐点：
    #    - 在外拐点到三角形中心的方向上
    #    - 距离三角形中心较近
    vec_to_triangle_center = triangle_center - np.array(outer_corner)
    vec_to_triangle_center_norm = vec_to_triangle_center / (np.linalg.norm(vec_to_triangle_center) + 1e-6)

    best_score = -1
    inner_corner = None

    for candidate in inner_candidates:
        vec_to_candidate = np.array(candidate) - np.array(outer_corner)
        dist_to_outer_corner = np.linalg.norm(vec_to_candidate)

        # 方向一致性
        vec_to_candidate_norm = vec_to_candidate / (dist_to_outer_corner + 1e-6)
        direction_alignment = np.dot(vec_to_candidate_norm, vec_to_triangle_center_norm)

        # 必须在正确的方向上
        if direction_alignment < 0.5:
            continue

        # 距离三角形中心
        dist_to_center = np.linalg.norm(np.array(candidate) - triangle_center)

        # 综合评分
        distance_score = 1.0 - (dist_to_center / (image_size / 2))
        score = direction_alignment * 0.5 + distance_score * 0.5

        if score > best_score:
            best_score = score
            inner_corner = candidate

    return inner_corner


def transform_keypoints(keypoints: list, M: np.ndarray):
    """
    将关键点通过透视变换矩阵映射到变换后的图像

    Args:
        keypoints: 原图中的关键点列表 [(x, y), ...]
        M: 透视变换矩阵

    Returns:
        变换后的关键点列表
    """
    keypoints_array = np.array(keypoints, dtype=np.float32).reshape(-1, 1, 2)
    transformed_keypoints = cv2.perspectiveTransform(keypoints_array, M)
    transformed_keypoints = transformed_keypoints.reshape(-1, 2).astype(np.int32)

    return [tuple(pt) for pt in transformed_keypoints]


def warp_l_shape_to_square(image: np.ndarray, contour: np.ndarray, square_size: int = 50, expand_ratio: float = 0.1):
    """
    将 L 型通过最小外接矩形透视变换成正方形，并返回变换后的轮廓点

    Args:
        image: 原始图像
        contour: 轮廓
        square_size: 输出正方形的边长
        expand_ratio: 向外扩展比例，默认 0.1（10%）

    Returns:
        tuple: (变换后的图像, 变换后的轮廓点, 透视变换矩阵)
    """
    # 1. 计算最小外接矩形
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.float32(box)

    # 2. 计算矩形中心
    center = rect[0]
    cx, cy = center

    # 3. 将矩形角点向外扩展
    expanded_box = []
    for pt in box:
        # 从中心指向角点的向量
        vec = np.array([pt[0] - cx, pt[1] - cy])
        # 扩展向量
        expanded_vec = vec * (1.0 + expand_ratio)
        # 新的角点
        expanded_pt = [cx + expanded_vec[0], cy + expanded_vec[1]]
        expanded_box.append(expanded_pt)

    expanded_box = np.float32(expanded_box)

    # 4. 源点：扩展后的矩形 4 个角点
    src_pts = expanded_box

    # 5. 目标点：正方形的 4 个角点
    dst_pts = np.float32([
        [0, 0],                          # 左上
        [square_size, 0],                # 右上
        [square_size, square_size],      # 右下
        [0, square_size]                 # 左下
    ])

    # 6. 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 7. 应用变换到图像
    warped = cv2.warpPerspective(image, M, (square_size, square_size))

    return warped, M


def visualize_perspective_transform(image_path: str, output_dir: str = "output/perspective"):
    """
    可视化 L 型透视变换效果

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    image = cv2.imread(image_path)

    # 提取红色区域
    mask = extract_red_mask(image)

    # 检测轮廓
    contours = find_contours(mask)
    contours = filter_contours_by_area(contours, min_area=200, max_area=10000)

    print(f"检测到 {len(contours)} 个轮廓")

    # 创建主可视化图像
    vis = image.copy()
    detected_count = 0
    warped_images = []

    for idx, contour in enumerate(contours):
        # 提取 L 型关键点
        result = find_l_shape_keypoints(contour)

        if result is None:
            # 未检测到 L 型，绘制灰色轮廓
            cv2.drawContours(vis, [contour], -1, (128, 128, 128), 1)
            continue

        detected_count += 1

        # 绘制原始轮廓（绿色）
        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)

        # 绘制 3 个关键点
        keypoints = result['keypoints']
        labels = ['端点1', '拐点', '端点2']
        colors = [(255, 0, 0), (0, 0, 255), (255, 0, 255)]  # 蓝、红、紫

        for i, (kp, label, color) in enumerate(zip(keypoints, labels, colors)):
            cv2.circle(vis, kp, 8, color, -1)
            cv2.circle(vis, kp, 8, (255, 255, 255), 2)
            cv2.putText(vis, label, (kp[0]+12, kp[1]-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 标注编号
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(vis, f"L#{detected_count}", (cx-25, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 透视变换成正方形（使用最小外接矩形）
        warped, M = warp_l_shape_to_square(image, contour, square_size=300, expand_ratio=0.1)

        # 将原图中的 3 个外轮廓关键点映射到变换后的图像
        keypoints = result['keypoints']
        outer_keypoints = transform_keypoints(keypoints, M)

        # 在变换后的图像中检测所有角点
        all_corners = detect_corners_in_warped(warped, quality_level=0.05, min_distance=30, max_corners=20)

        # 根据外轮廓角点推算内拐点（传入图像用于红色区域检查）
        inner_corner = find_inner_corners_from_outer(outer_keypoints, all_corners, warped, image_size=300)

        # 在变换后的图像上绘制角点
        warped_vis = warped.copy()

        # 绘制外轮廓角点（绿色）
        labels = ['端点1', '拐点', '端点2']
        for i, (kp, label) in enumerate(zip(outer_keypoints, labels)):
            cv2.circle(warped_vis, kp, 8, (0, 255, 0), -1)
            cv2.circle(warped_vis, kp, 8, (255, 255, 255), 2)
            cv2.putText(warped_vis, label, (kp[0]+12, kp[1]-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 绘制内拐点（红色）
        if inner_corner is not None:
            cv2.circle(warped_vis, inner_corner, 8, (0, 0, 255), -1)
            cv2.circle(warped_vis, inner_corner, 8, (255, 255, 255), 2)
            cv2.putText(warped_vis, '内拐点', (inner_corner[0]+12, inner_corner[1]+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        warped_images.append((detected_count, warped_vis))

        # 保存单个变换结果
        warped_path = f"{output_dir}/warped_{detected_count}.jpg"
        cv2.imwrite(warped_path, warped_vis)

        # 保存红色二值化图片用于调试
        red_mask_debug = extract_red_mask(warped)
        mask_path = f"{output_dir}/mask_{detected_count}.jpg"
        cv2.imwrite(mask_path, red_mask_debug)

        # 打印角点坐标
        print(f"L#{detected_count}:")
        print(f"  外轮廓角点: {outer_keypoints}")
        if inner_corner is not None:
            print(f"  内拐点: {inner_corner}")
        else:
            print(f"  内拐点: 未找到")

    # 保存主可视化图像
    basename = os.path.basename(image_path).split('.')[0]
    vis_path = f"{output_dir}/{basename}_detected.jpg"
    cv2.imwrite(vis_path, vis)

    print(f"\n=== 检测结果 ===")
    print(f"检测到 {detected_count} 个 L 型灯条")
    print(f"主可视化图像: {vis_path}")

    # 创建拼接图：显示所有变换后的正方形
    if len(warped_images) > 0:
        # 每行显示 4 个
        cols = 4
        rows = (len(warped_images) + cols - 1) // cols

        # 创建拼接画布
        canvas_height = rows * 300
        canvas_width = cols * 300
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        for i, (num, warped) in enumerate(warped_images):
            row = i // cols
            col = i % cols
            y_start = row * 300
            x_start = col * 300

            canvas[y_start:y_start+300, x_start:x_start+300] = warped

            # 添加编号
            cv2.putText(canvas, f"L#{num}", (x_start+10, y_start+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # 保存拼接图
        grid_path = f"{output_dir}/{basename}_warped_grid.jpg"
        cv2.imwrite(grid_path, canvas)
        print(f"变换拼接图: {grid_path}")

    print(f"\n所有变换结果已保存到: {output_dir}/")


def test_new_corner_detector(image_path: str):
    """
    测试新的角点检测算法（基于轮廓角度分析）

    Args:
        image_path: 输入图像路径
    """
    print(f"\n{'='*60}")
    print(f"测试新角点检测算法: {image_path}")
    print(f"{'='*60}\n")

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

    print(f"检测到 {len(contours)} 个候选轮廓")

    # 创建输出目录
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = f"output/{basename}_new_corner"
    os.makedirs(output_dir, exist_ok=True)

    # 统计
    success_count = 0
    total_count = len(contours)

    for i, contour in enumerate(contours):
        print(f"\n处理轮廓 #{i+1}/{total_count}")

        # 旧算法：基于最小外接矩形
        old_result = find_l_shape_keypoints(contour)
        if old_result is None:
            print(f"  旧算法: 失败")
            continue

        old_keypoints = old_result['keypoints']

        # 透视变换（使用仿射变换，因为只有3个点）
        src_pts = np.float32(old_keypoints)
        dst_pts = np.float32([[0, 0], [0, 300], [300, 300]])
        matrix = cv2.getAffineTransform(src_pts, dst_pts)
        warped = cv2.warpAffine(image, matrix, (300, 300))

        # 在变换后的图像上提取轮廓
        warped_mask = extract_red_mask(warped)
        warped_contours = find_contours(warped_mask)

        if len(warped_contours) == 0:
            print(f"  变换后无轮廓")
            continue

        # 选择最大轮廓
        warped_contour = max(warped_contours, key=cv2.contourArea)

        # 新算法：基于轮廓角度分析
        new_result = extract_l_shape_corners(warped_contour, warped, image_size=300)

        if new_result is None:
            print(f"  新算法: 失败")
            continue

        success_count += 1

        # 提取结果
        outer_keypoints = new_result['outer_keypoints']
        inner_corner = new_result['inner_corner']
        angles = new_result['angles']
        gradient = new_result['gradient']
        segments = new_result['segments']

        print(f"  新算法: 成功")
        print(f"    外轮廓角点: {len(outer_keypoints)} 个")
        print(f"    内拐点: {'找到' if inner_corner else '未找到'}")
        print(f"    单调区间: {len(segments)} 个")

        # 可视化
        vis = warped.copy()

        # 绘制轮廓
        cv2.drawContours(vis, [warped_contour], -1, (0, 255, 0), 2)

        # 绘制外轮廓角点
        for idx, pt in enumerate(outer_keypoints):
            cv2.circle(vis, pt, 5, (0, 0, 255), -1)
            cv2.putText(vis, f"O{idx+1}", (pt[0]+10, pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 绘制内拐点
        if inner_corner:
            cv2.circle(vis, inner_corner, 5, (255, 0, 255), -1)
            cv2.putText(vis, "Inner", (inner_corner[0]+10, inner_corner[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # 保存结果
        output_path = f"{output_dir}/L{i+1:02d}_corners.jpg"
        cv2.imwrite(output_path, vis)

        # 可视化角度和梯度曲线
        fig_path = f"{output_dir}/L{i+1:02d}_analysis.jpg"
        visualize_angle_gradient(angles, gradient, segments, fig_path)

    print(f"\n{'='*60}")
    print(f"测试完成: {success_count}/{total_count} 成功")
    print(f"结果保存到: {output_dir}/")
    print(f"{'='*60}\n")


def visualize_angle_gradient(angles: np.ndarray, gradient: np.ndarray,
                              segments: list, output_path: str):
    """
    可视化角度和梯度曲线

    Args:
        angles: 角度数组 (1, N)
        gradient: 梯度数组 (1, N)
        segments: 单调区间列表
        output_path: 输出路径
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    angles_flat = angles.flatten()
    gradient_flat = gradient.flatten()
    n = len(angles_flat)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 绘制角度曲线
    ax1.plot(angles_flat, 'b-', linewidth=1)
    ax1.set_xlabel('Contour Point Index')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Contour Angle Array')
    ax1.grid(True, alpha=0.3)

    # 标记单调区间
    for start, end in segments:
        ax1.axvspan(start, end, alpha=0.2, color='green')

    # 绘制梯度曲线
    ax2.plot(gradient_flat, 'r-', linewidth=1)
    ax2.axhline(y=3.0, color='g', linestyle='--', label='Threshold (+3)')
    ax2.axhline(y=-3.0, color='g', linestyle='--', label='Threshold (-3)')
    ax2.set_xlabel('Contour Point Index')
    ax2.set_ylabel('Gradient')
    ax2.set_title('Angle Gradient')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 标记单调区间
    for start, end in segments:
        ax2.axvspan(start, end, alpha=0.2, color='green')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 test_perspective_transform.py <image_path> [--new]")
        print("  --new: 使用新的角点检测算法")
        sys.exit(1)

    image_path = sys.argv[1]

    # 如果路径不是绝对路径，添加 data/ 前缀
    if not os.path.isabs(image_path) and not os.path.exists(image_path):
        image_path = os.path.join('data', image_path)

    # 检查是否使用新算法
    use_new = '--new' in sys.argv

    if use_new:
        test_new_corner_detector(image_path)
    else:
        visualize_perspective_transform(image_path)
