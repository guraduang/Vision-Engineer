#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import os

def extract_red_mask(image):
    """提取红色区域"""
    b, g, r = cv2.split(image)
    red_diff = cv2.subtract(r, b)
    _, mask = cv2.threshold(red_diff, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def get_chain_code(contour):
    """
    计算轮廓的链码
    8方向链码：0=右, 1=右上, 2=上, 3=左上, 4=左, 5=左下, 6=下, 7=右下
    """
    if len(contour) < 2:
        return []

    chain = []
    for i in range(len(contour)):
        pt1 = contour[i][0]
        pt2 = contour[(i + 1) % len(contour)][0]

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]

        # 计算方向
        if dx == 0 and dy == 0:
            continue

        angle = np.arctan2(-dy, dx)  # OpenCV y轴向下
        angle_deg = np.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360

        # 映射到8方向
        direction = int((angle_deg + 22.5) / 45) % 8
        chain.append(direction)

    return chain

def normalize_chain_code(chain):
    """
    归一化链码（旋转不变性）
    找到字典序最小的旋转
    """
    if not chain:
        return []

    n = len(chain)
    min_chain = chain

    for start in range(n):
        rotated = chain[start:] + chain[:start]
        if rotated < min_chain:
            min_chain = rotated

    return min_chain

def simplify_chain_code(chain, merge_threshold=3):
    """
    简化链码：合并连续相同方向
    返回：[(direction, count), ...]
    """
    if not chain:
        return []

    simplified = []
    current_dir = chain[0]
    count = 1

    for i in range(1, len(chain)):
        if chain[i] == current_dir:
            count += 1
        else:
            if count >= merge_threshold:
                simplified.append((current_dir, count))
            current_dir = chain[i]
            count = 1

    if count >= merge_threshold:
        simplified.append((current_dir, count))

    return simplified

def warp_to_frontal(image, contour):
    """
    透视变换：将轮廓校正到正面视角
    假设L型是矩形的组合，尝试找到主方向并校正
    """
    # 获取最小外接矩形
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 计算目标矩形大小
    width = int(rect[1][0])
    height = int(rect[1][1])

    if width == 0 or height == 0:
        return None, None

    # 目标点（正面视角）
    dst_pts = np.array([
        [0, height],
        [0, 0],
        [width, 0],
        [width, height]
    ], dtype=np.float32)

    # 源点
    src_pts = box.astype(np.float32)

    # 透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 变换图像
    warped = cv2.warpPerspective(image, M, (width, height))

    # 变换轮廓
    contour_reshaped = contour.reshape(-1, 1, 2).astype(np.float32)
    warped_contour = cv2.perspectiveTransform(contour_reshaped, M)

    return warped, warped_contour

def match_l_shape_chain(chain, template=[0, 2, 0, 6, 4, 2], tolerance=1):
    """
    匹配L型链码模板
    template: 标准L型链码 [0, 2, 0, 6, 4, 2]
    tolerance: 允许的方向偏差

    由于透视变换，实际链码可能不完全匹配模板
    我们检查：
    1. 是否有4个主方向（0,2,4,6）
    2. 方向变化次数是否合理（4-8次）
    """
    if len(chain) < 4 or len(chain) > 8:
        return False, 0

    # 统计主方向（0,2,4,6）和对角方向（1,3,5,7）
    main_dirs = [d for d in chain if d in [0, 2, 4, 6]]
    diag_dirs = [d for d in chain if d in [1, 3, 5, 7]]

    # L型应该主要由主方向组成
    if len(main_dirs) < len(chain) * 0.5:  # 至少50%是主方向
        return False, 0

    # 检查是否包含至少3个不同的主方向
    unique_main = set(main_dirs)
    if len(unique_main) < 3:
        return False, 0

    # 计算匹配分数
    score = 0
    # 有4个主方向更好
    if len(unique_main) >= 4:
        score += 0.3
    elif len(unique_main) >= 3:
        score += 0.2

    # 长度合适
    if 5 <= len(chain) <= 7:
        score += 0.3
    elif 4 <= len(chain) <= 8:
        score += 0.2

    # 主方向占比高
    main_ratio = len(main_dirs) / len(chain)
    score += main_ratio * 0.4

    return score >= 0.5, score

def is_l_shape_chaincode(contour, image, epsilon_factor=0.02):
    """
    使用链码判断是否为L型
    """
    # 轮廓近似
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 基本筛选
    if len(approx) < 4 or len(approx) > 8:
        return False, None, {}

    # 凸包比
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    convex_ratio = contour_area / hull_area if hull_area > 0 else 0

    if convex_ratio >= 0.85:  # L型应该是凹的
        return False, None, {}

    # 面积筛选
    area = cv2.contourArea(contour)
    if area < 100:
        return False, None, {}

    # 提取链码
    chain = get_chain_code(approx)
    if len(chain) < 4:
        return False, None, {}

    # 简化链码
    simplified = simplify_chain_code(chain, merge_threshold=1)
    simplified_dirs = [d for d, c in simplified]

    # 归一化
    normalized = normalize_chain_code(simplified_dirs)

    # 匹配L型模板
    is_match, match_ratio = match_l_shape_chain(normalized)

    # 检查边长约束
    lengths = []
    for i in range(len(approx)):
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % len(approx)][0]
        length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        lengths.append(length)

    # 寻找相似边对
    def edges_match(len1, len2, tolerance=0.20):  # 降低到20%
        ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        return ratio >= (1 - tolerance)

    pair_matches = 0
    matched_pairs = []
    for i in range(len(lengths)):
        for j in range(i + 1, len(lengths)):
            if edges_match(lengths[i], lengths[j]):
                pair_matches += 1
                matched_pairs.append((i, j, lengths[i], lengths[j]))

    info = {
        'approx': approx,
        'chain': chain,
        'simplified': simplified,
        'normalized': normalized,
        'match_ratio': match_ratio,
        'convex_ratio': convex_ratio,
        'lengths': lengths,
        'pair_matches': pair_matches,
        'matched_pairs': matched_pairs
    }

    if not is_match:
        return False, None, info

    if pair_matches < 2:
        return False, None, info

    return True, approx, info

def process_image(image_path):
    """处理单张图像"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return

    print(f"=== Processing {os.path.basename(image_path)} ===")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # 提取红色mask
    mask = extract_red_mask(image)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours: {len(contours)}")

    # 筛选候选轮廓
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        candidates.append(contour)

    print(f"Candidate contours: {len(candidates)}")

    # 检测L型
    l_shapes = []
    candidate_count = 0
    for contour in candidates:
        candidate_count += 1
        for epsilon_factor in [0.02, 0.015, 0.025, 0.03]:
            is_l, approx, info = is_l_shape_chaincode(contour, image, epsilon_factor)

            # 调试输出
            if info:
                print(f"\nCandidate #{candidate_count} (epsilon={epsilon_factor}):")
                print(f"  Points: {len(info.get('approx', []))}")
                print(f"  Convex ratio: {info.get('convex_ratio', 0):.3f}")
                print(f"  Chain: {info.get('chain', [])}")
                print(f"  Simplified: {info.get('simplified', [])}")
                print(f"  Normalized: {info.get('normalized', [])}")
                print(f"  Match ratio: {info.get('match_ratio', 0):.3f}")
                print(f"  Lengths: {[f'{l:.1f}' for l in info.get('lengths', [])]}")
                print(f"  Pair matches: {info.get('pair_matches', 0)}")
                if info.get('matched_pairs'):
                    print(f"  Matched pairs: {[(i, j, f'{l1:.1f}', f'{l2:.1f}') for i, j, l1, l2 in info.get('matched_pairs', [])]}")
                print(f"  Is L-shape: {is_l}")

            if is_l:
                l_shapes.append({
                    'contour': contour,
                    'approx': approx,
                    'info': info
                })
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

    print(f"\n=== Detection Results ===")
    print(f"Total L-shapes detected: {len(unique_l_shapes)}\n")

    # 绘制结果
    result = image.copy()
    for idx, shape in enumerate(unique_l_shapes):
        contour = shape['contour']
        approx = shape['approx']
        info = shape['info']

        # 绘制轮廓
        cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)

        # 绘制中心点
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(result, f"#{idx+1}", (cx+10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 打印信息
        print(f"L-shape #{idx+1}:")
        print(f"  Chain code: {info['chain']}")
        print(f"  Simplified: {info['simplified']}")
        print(f"  Normalized: {info['normalized']}")
        print(f"  Match ratio: {info['match_ratio']:.2f}")
        print(f"  Convex ratio: {info['convex_ratio']:.3f}")
        print(f"  Pair matches: {info['pair_matches']}")
        print(f"  Lengths: {[f'{l:.1f}' for l in info['lengths']]}")
        print()

    # 保存结果
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_chaincode.jpg")
    cv2.imwrite(output_path, result)
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_chaincode_perspective.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    process_image(image_path)
