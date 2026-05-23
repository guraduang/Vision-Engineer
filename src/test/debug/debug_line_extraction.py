#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于角度梯度的直线段提取调试工具
参考 RM 视觉核心算法：角度曲线 → 梯度计算 → 直线段提取
"""

import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from perception import (
    compute_warped_area_ratio,
    extract_red_mask_from_gray_contours,
    find_contours,
    filter_contours_by_area,
    filter_contours_by_geometry,
)
from calculation.contour_corner_detector import (
    compute_contour_angles,
    compute_angle_gradient
)
from calculation.perspective import warp_to_frontal, warp_to_frontal_with_matrix


@dataclass
class Line:
    """直线段结构"""
    start_idx: int      # 起始索引
    end_idx: int        # 结束索引
    start_angle: float  # 起始角度
    end_angle: float    # 结束角度
    avg_angle: float    # 平均角度
    center_x: float     # 中心点 x
    center_y: float     # 中心点 y
    length: int         # 线段长度（点数）


# 侧视/多灯条：过宽的 warped 占比与过大原轮廓易误检；warp 后红掩膜为空时用小轮廓恢复。
LINE_EXTRACT_CANDIDATE_MIN_AREA = 100
LINE_EXTRACT_CANDIDATE_MAX_AREA = 420
LINE_EXTRACT_MAX_AREA_SCALE_BY_MEDIAN = 4.0
LINE_EXTRACT_WARPED_AREA_MIN = 0.10
LINE_EXTRACT_WARPED_AREA_MAX = 0.585
LINE_EXTRACT_RECOVER_MIN_AREA = 180
LINE_EXTRACT_RECOVER_MIN_POINTS = 12
LINE_EXTRACT_LONG_EDGE_MIN_POINTS = 45
LINE_EXTRACT_STEEP_GRAD_ABS = 30.0
LINE_EXTRACT_STEEP_GRAD_Q95 = 40.0
LINE_EXTRACT_STEEP_RATIO_MAX = 0.20
LINE_EXTRACT_CURVE_MEAN_DIST_MAX = 7.5
LINE_EXTRACT_CURVE_P90_DIST_MAX = 14.0


def _resolve_writable_output_dir(basename: str) -> str:
    """
    选择可写输出目录：
    1) 优先 output/debug_line_segments/<basename>
    2) 不可写时回退 /tmp/vision_debug_line_segments/<basename>
    """
    preferred = os.path.join("output", "debug_line_segments", basename)
    fallback = os.path.join("/tmp", "vision_debug_line_segments", basename)

    def _ensure_writable(path: str) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, ".write_probe.tmp")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            return True
        except Exception:
            return False

    if _ensure_writable(preferred):
        return preferred
    _ensure_writable(fallback)
    print(f"[warn] output dir not writable, fallback to: {fallback}")
    return fallback


def filter_candidates_for_line_extraction(
        image: np.ndarray,
        contours_after_geometry: List[np.ndarray]) -> List[np.ndarray]:
    """透视矫正后面积占比 + 原图面积窗；warp 后无有效红轮廓时按小目标恢复。"""
    out: List[np.ndarray] = []
    if not contours_after_geometry:
        return out

    # 使用当前帧几何候选面积中位数得到自适应上限，避免固定 420 误杀远近变化明显的真目标。
    areas = [float(cv2.contourArea(c)) for c in contours_after_geometry]
    area_median = float(np.median(np.asarray(areas, dtype=np.float64)))
    adaptive_max_area = max(
        LINE_EXTRACT_CANDIDATE_MAX_AREA,
        area_median * LINE_EXTRACT_MAX_AREA_SCALE_BY_MEDIAN,
    )

    warp_fn = lambda img, cnt: warp_to_frontal(img, cnt)[0]
    for cnt in contours_after_geometry:
        a = float(cv2.contourArea(cnt))
        if a < LINE_EXTRACT_CANDIDATE_MIN_AREA or a > adaptive_max_area:
            continue
        ar = compute_warped_area_ratio(
            image,
            cnt,
            warp_fn,
            extract_red_mask_from_gray_contours,
            warped_min_area=50.0,
        )
        if ar is not None:
            if LINE_EXTRACT_WARPED_AREA_MIN <= ar <= LINE_EXTRACT_WARPED_AREA_MAX:
                out.append(cnt)
        elif a >= LINE_EXTRACT_RECOVER_MIN_AREA and len(cnt) >= LINE_EXTRACT_RECOVER_MIN_POINTS:
            out.append(cnt)
    return out


def validate_l_shape_edges(lines):
    """
    验证直线段是否符合 L 型的几何约束

    L 型应该有 4 条边：
    - 两两角度差 ≈ 180° (对边平行)
    - 两两角度差 ≈ 90° (相邻边垂直)

    Args:
        lines: 直线段列表

    Returns:
        (is_valid, info) - 是否符合L型，详细信息
    """
    if len(lines) < 2:
        return False, "线段数量 < 2"

    # 提取所有角度
    angles = [line.avg_angle for line in lines]

    # 归一化角度到 [-180, 180]
    angles_norm = []
    for angle in angles:
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        angles_norm.append(angle)

    # 如果只有 2 条线段，检查是否垂直
    if len(lines) == 2:
        angle_diff = abs(angles_norm[0] - angles_norm[1])
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        is_perpendicular = abs(angle_diff - 90) < 25  # 允许 25° 误差

        if is_perpendicular:
            return True, f"2条边垂直 (角度差={angle_diff:.1f}°)"
        else:
            return False, f"2条边不垂直 (角度差={angle_diff:.1f}°)"

    # 如果有 3-4 条线段，尝试找到 2 对平行边
    if len(lines) >= 3:
        # 按角度分组（角度相近的为一组）
        angle_groups = []
        used = set()

        for i, angle1 in enumerate(angles_norm):
            if i in used:
                continue

            group = [i]
            for j, angle2 in enumerate(angles_norm):
                if i == j or j in used:
                    continue

                # 检查是否平行（角度差 ≈ 0° 或 180°）
                angle_diff = abs(angle1 - angle2)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                if angle_diff < 25 or abs(angle_diff - 180) < 25:
                    group.append(j)
                    used.add(j)

            if len(group) > 0:
                angle_groups.append(group)
                used.add(i)

        # 检查是否有 2 组，且两组之间垂直
        if len(angle_groups) >= 2:
            group1_angle = np.mean([angles_norm[i] for i in angle_groups[0]])
            group2_angle = np.mean([angles_norm[i] for i in angle_groups[1]])

            angle_diff = abs(group1_angle - group2_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            is_perpendicular = abs(angle_diff - 90) < 25  # 允许 25° 误差

            if is_perpendicular:
                return True, f"找到2组平行边，组间垂直 (角度差={angle_diff:.1f}°)"
            else:
                return False, f"找到2组边，但不垂直 (角度差={angle_diff:.1f}°)"

        return False, f"无法分组为平行边"

    return False, "线段数量不足"


def _normalize_deg_360(angle: float) -> float:
    """角度归一化到 [0, 360)。"""
    a = float(angle) % 360.0
    if a < 0:
        a += 360.0
    return a


def _circular_diff_deg(a: float, b: float) -> float:
    """圆周角差，返回 [0, 180]。"""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def line_angle_to_chaincode8(angle_deg: float) -> int:
    """
    将有向直线角度映射到八方向链码（0~7）。
    约定（y 轴向上）：
      0=右, 1=右上, 2=上, 3=左上, 4=左, 5=左下, 6=下, 7=右下
    """
    a = _normalize_deg_360(angle_deg)
    candidates = [
        (0.0, 0),
        (45.0, 1),
        (90.0, 2),
        (135.0, 3),
        (180.0, 4),
        (225.0, 5),
        (270.0, 6),
        (315.0, 7),
    ]
    best_code = 0
    best_diff = 1e9
    for ref_deg, code in candidates:
        d = _circular_diff_deg(a, ref_deg)
        if d < best_diff:
            best_diff = d
            best_code = code
    return best_code


def sort_lines_in_contour_order(lines: List[Line], contour_len: int) -> List[Line]:
    """
    按轮廓访问顺序排序线段。
    对跨接缝线段（start_idx > end_idx）将其视为最早出现，避免顺序错乱。
    """
    def _key(line: Line) -> int:
        if line.start_idx <= line.end_idx:
            return line.start_idx
        return line.start_idx - contour_len

    return sorted(lines, key=_key)


def _stabilize_chain_segments(
        chain: List[int],
        lengths: List[int],
        target_len: int = 6,
        short_ratio: float = 0.35,
        min_short_len: int = 3) -> Tuple[List[int], List[int]]:
    """
    对四方向链码做环形稳态化：
    1) 先合并相邻同方向段；
    2) 若段数仍多于 target_len，优先吞并短噪声段（接缝碎片/毛刺短线）。
    """
    if not chain or len(chain) != len(lengths):
        return list(chain), list(lengths)

    c = [int(v) for v in chain]
    l = [int(v) for v in lengths]

    def merge_adjacent_same_direction() -> None:
        if len(c) < 2:
            return
        changed = True
        while changed and len(c) >= 2:
            changed = False
            n_local = len(c)
            for i in range(n_local):
                j = (i + 1) % n_local
                if c[i] == c[j]:
                    l[i] += l[j]
                    del c[j]
                    del l[j]
                    changed = True
                    break

    merge_adjacent_same_direction()

    while len(c) > target_len and len(c) >= 3:
        med_len = float(np.median(l)) if l else 0.0
        short_thr = max(min_short_len, int(round(short_ratio * med_len)))

        short_indices = [i for i, seg_len in enumerate(l) if seg_len <= short_thr]
        if not short_indices:
            short_indices = [int(np.argmin(l))]

        i = min(short_indices, key=lambda idx: l[idx])
        n_local = len(c)
        prev_i = (i - 1) % n_local
        next_i = (i + 1) % n_local

        if c[prev_i] == c[next_i] and prev_i != next_i:
            merged_len = l[prev_i] + l[i] + l[next_i]
            c[prev_i] = c[prev_i]
            l[prev_i] = merged_len

            for rm in sorted([i, next_i], reverse=True):
                del c[rm]
                del l[rm]
        else:
            merge_to_prev = l[prev_i] >= l[next_i]
            if merge_to_prev:
                l[prev_i] += l[i]
                del c[i]
                del l[i]
            else:
                l[next_i] += l[i]
                del c[i]
                del l[i]

        merge_adjacent_same_direction()

    return c, l


def weighted_chaincode_match(
        chain: List[int],
        lengths: List[int],
        template: List[int] = None,
        heavy_weight: float = 2.0) -> Tuple[float, List[int], List[float], List[int], List[int], List[int]]:
    """
    对 6 段四方向链码做加权模板匹配：
    - 最长的 4 条线权重为 heavy_weight
    - 其余线权重为 1.0
    - 在模板循环位移与反向模板（方向反转）中取最佳匹配

    Returns:
        (best_score, best_template_variant, weights, longest4_indices, chain_used, lengths_used)
    """
    if template is None:
        template = [0, 6, 4, 6, 4, 2]

    def rotations(seq: List[int]) -> List[List[int]]:
        return [seq[k:] + seq[:k] for k in range(len(seq))]

    def _evaluate(chain_eval: List[int], lengths_eval: List[int]) -> Tuple[float, List[int], List[float], List[int]]:
        if len(chain_eval) != len(template):
            return 0.0, template, [1.0] * len(chain_eval), []

        n = len(chain_eval)
        order = sorted(range(n), key=lambda i: lengths_eval[i], reverse=True)
        longest4 = set(order[:4])
        weights = [heavy_weight if i in longest4 else 1.0 for i in range(n)]
        total_weight = float(sum(weights)) if weights else 1.0

        # 反向遍历时方向取反（+4 mod 8）
        reversed_template = [((d + 4) % 8) for d in template[::-1]]
        variants = rotations(template) + rotations(reversed_template)

        best_score_local = -1.0
        best_variant_local = variants[0]
        for var in variants:
            score = 0.0
            for i, c in enumerate(chain_eval):
                if c == var[i]:
                    score += weights[i]
            score /= total_weight
            if score > best_score_local:
                best_score_local = score
                best_variant_local = var
        return best_score_local, best_variant_local, weights, sorted(longest4)

    chain_raw = [int(v) for v in chain]
    lengths_raw = [int(v) for v in lengths]
    raw_score, raw_variant, raw_weights, raw_longest4 = _evaluate(chain_raw, lengths_raw)

    chain_stable, lengths_stable = _stabilize_chain_segments(chain_raw, lengths_raw, target_len=len(template))
    stable_score, stable_variant, stable_weights, stable_longest4 = _evaluate(chain_stable, lengths_stable)

    if stable_score > raw_score:
        return stable_score, stable_variant, stable_weights, stable_longest4, chain_stable, lengths_stable
    return raw_score, raw_variant, raw_weights, raw_longest4, chain_raw, lengths_raw


def _angle_diff_deg(a: float, b: float) -> float:
    """两角度之差，归一化到 [0, 180]。"""
    d = abs(float(a) - float(b))
    if d > 180:
        d = 360.0 - d
    return d


def _line_orientation_diff_deg(a: float, b: float) -> float:
    """无向直线方向差，按 180 度周期比较，结果归一化到 [0, 90]。"""
    aa = float(a) % 180.0
    bb = float(b) % 180.0
    d = abs(aa - bb)
    if d > 90.0:
        d = 180.0 - d
    return d


def _wrap_index_dist(i: int, j: int, n: int) -> int:
    """闭合轮廓上两下标的最短弧长（步数）。"""
    d = abs(i - j)
    return min(d, n - d)


def _find_corner_roll_index(gradient_flat: np.ndarray) -> int:
    """
    取 |角度梯度| 的（闭合序列）局部极大中最大的峰，对应轮廓上最显著的角点；
    np.roll 后该点落在下标 0，闭合接缝远离长直边中点。
    """
    g = np.abs(np.asarray(gradient_flat, dtype=np.float64).ravel())
    n = len(g)
    if n < 3:
        return 0
    peaks: List[Tuple[int, float]] = []
    for i in range(n):
        if g[i] >= g[(i - 1) % n] and g[i] >= g[(i + 1) % n]:
            peaks.append((i, float(g[i])))
    if not peaks:
        return int(np.argmax(g))
    peaks.sort(key=lambda t: -t[1])
    return int(peaks[0][0])


def _circular_moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """闭合序列滑动平均，减小角度梯度抖动导致的碎片段。"""
    sig = np.asarray(signal, dtype=np.float64).ravel()
    n = len(sig)
    if n == 0:
        return sig
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w <= 1 or n < 3:
        return sig

    pad = w // 2
    ext = np.concatenate([sig[-pad:], sig, sig[:pad]])
    ker = np.ones(w, dtype=np.float64) / float(w)
    sm = np.convolve(ext, ker, mode='valid')
    return sm[:n]


def _bridge_small_straight_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """在闭合一维 straight-mask 上填补短暂断裂（False 缺口）。"""
    arr = np.asarray(mask, dtype=np.uint8).ravel()
    n = len(arr)
    if n == 0 or max_gap <= 0:
        return arr.astype(bool)

    g = int(max_gap)
    ext = np.concatenate([arr[-g:], arr, arr[:g]])
    kernel = np.ones((1, 2 * g + 1), dtype=np.uint8)
    closed = cv2.morphologyEx((ext.reshape(1, -1) * 255), cv2.MORPH_CLOSE, kernel)
    bridged = closed[0, g:g + n] > 0
    return bridged


def _mean_line_orientation_deg(angles_deg: np.ndarray) -> float:
    """无向直线方向均值（180 度周期），避免 ±180 接缝导致均值跳变。"""
    ang = np.asarray(angles_deg, dtype=np.float64).ravel()
    if len(ang) == 0:
        return 0.0
    rad2 = np.deg2rad(2.0 * ang)
    s = float(np.mean(np.sin(rad2)))
    c = float(np.mean(np.cos(rad2)))
    mean_half = 0.5 * np.arctan2(s, c)
    return float(np.rad2deg(mean_half))


def _choose_best_line_result(
        candidates: List[Tuple[np.ndarray, np.ndarray, List[Line], float, bool]],
        target_line_count: int = 6) -> Tuple[np.ndarray, np.ndarray, List[Line], float, bool]:
    """
    在多套提线结果中自动选择最适合链码的版本（优先 6 段）。
    评分优先级：
      1) 段数与 target_line_count 的差距
      2) 极短段数量（<=4 点，越少越好）
      3) 最长4段总长度（越大越好）
      4) 总长度（越大越好）
    """
    if not candidates:
        return None, None, [], 0.0, False

    def quality(item: Tuple[np.ndarray, np.ndarray, List[Line], float, bool]) -> Tuple[int, int, int, int]:
        _angles, _grad, lines, _thr, _rolled = item
        n = len(lines)
        if n <= 0:
            return (10**9, 10**9, 10**9, 10**9)
        lengths = sorted([int(L.length) for L in lines], reverse=True)
        short_cnt = sum(1 for v in lengths if v <= 4)
        top4_sum = int(sum(lengths[:4]))
        total_sum = int(sum(lengths))
        return (abs(n - target_line_count), short_cnt, -top4_sum, -total_sum)

    return min(candidates, key=quality)


def approx_poly_dp_n_vertices(
        contour: np.ndarray,
        n_vertices: int = 6,
        max_iter: int = 64) -> Optional[np.ndarray]:
    """二分搜索 epsilon，使 approxPolyDP 返回恰好 n_vertices 个顶点。"""
    if contour is None or len(contour) < max(10, n_vertices * 2):
        return None
    peri = cv2.arcLength(contour, True)
    if peri <= 1e-6:
        return None

    lo, hi = 1e-9 * peri, 0.5 * peri
    best = None
    best_diff = 10**9
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        approx = cv2.approxPolyDP(contour, mid, True)
        c = len(approx)
        diff = abs(c - n_vertices)
        if diff < best_diff:
            best_diff = diff
            best = approx
        if c == n_vertices:
            return approx
        if c > n_vertices:
            lo = mid
        else:
            hi = mid
    return best if best is not None and len(best) == n_vertices else None


def _segment_indices(start_idx: int, end_idx: int, n: int) -> np.ndarray:
    """闭合轮廓上从 start 到 end 的索引序列（包含两端）。"""
    if start_idx <= end_idx:
        return np.arange(start_idx, end_idx + 1, dtype=np.int32)
    return np.concatenate([
        np.arange(start_idx, n, dtype=np.int32),
        np.arange(0, end_idx + 1, dtype=np.int32),
    ])


def _fit_line_for_segment(
        contour_pts: np.ndarray,
        contour_grad_metric: np.ndarray,
        seg_idx: np.ndarray,
        start_idx: int,
        end_idx: int,
        trim_ratio: float = 0.08,
        grad_keep_quantile: float = 0.70,
        min_keep_ratio: float = 0.45) -> Tuple[Optional[Line], Dict]:
    """对一段轮廓点做梯度引导拟合：先剔除高梯度点，再拟合直线。"""
    if seg_idx is None or len(seg_idx) < 2:
        return None, {}

    pts = contour_pts[seg_idx]
    nseg = len(pts)
    # 短边少 trim，避免拟合点被吃光
    if nseg <= 8:
        trim = 0
    else:
        trim = int(round(nseg * float(trim_ratio)))
        trim = min(trim, max(0, (nseg - 2) // 2))

    if trim * 2 < nseg - 1:
        inner = seg_idx[trim:len(seg_idx) - trim]
        fit_idx = np.array(inner, dtype=np.int32)
        fit_pts = contour_pts[fit_idx]
    else:
        fit_idx = np.array(seg_idx, dtype=np.int32)
        fit_pts = pts.copy()

    if len(fit_idx) < 2:
        fit_idx = np.array(seg_idx, dtype=np.int32)
        fit_pts = contour_pts[fit_idx]

    # 梯度筛点：丢弃段内高梯度（角点过渡区/毛刺）点，仅保留较平滑点拟合
    grad = np.abs(contour_grad_metric[fit_idx]).astype(np.float64)
    steep_ratio = float(np.mean(grad >= LINE_EXTRACT_STEEP_GRAD_ABS)) if len(grad) > 0 else 0.0
    steep_q95 = float(np.quantile(grad, 0.95)) if len(grad) > 0 else 0.0

    # 先在整段上做一次粗拟合，若残差过大，说明该段整体是弯折/锯齿，不是近似直线。
    seg_pts = contour_pts[np.array(seg_idx, dtype=np.int32)]
    seg_pts_cv = seg_pts.reshape(-1, 1, 2).astype(np.float32)
    svx, svy, sx0, sy0 = cv2.fitLine(
        seg_pts_cv, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    sdx = seg_pts[:, 0] - float(sx0)
    sdy = seg_pts[:, 1] - float(sy0)
    seg_dist = np.abs(float(svy) * sdx - float(svx) * sdy)
    seg_mean_dist = float(np.mean(seg_dist)) if len(seg_dist) > 0 else 0.0
    seg_p90_dist = float(np.quantile(seg_dist, 0.90)) if len(seg_dist) > 0 else 0.0

    # 长边上若“高梯度过陡且出现过多”，判为非近似直线，直接拒绝该边。
    if (len(seg_idx) >= LINE_EXTRACT_LONG_EDGE_MIN_POINTS
            and steep_q95 >= LINE_EXTRACT_STEEP_GRAD_Q95
            and steep_ratio >= LINE_EXTRACT_STEEP_RATIO_MAX):
        return None, {
            'segment_indices': np.array(seg_idx, dtype=np.int32),
            'fit_indices': np.array(fit_idx, dtype=np.int32),
            'num_segment_points': int(len(seg_idx)),
            'num_fit_points': int(len(fit_idx)),
            'steep_ratio': float(steep_ratio),
            'steep_q95': float(steep_q95),
            'segment_mean_dist': float(seg_mean_dist),
            'segment_p90_dist': float(seg_p90_dist),
            'reject_reason': (
                f'long edge non-line: steep_ratio={steep_ratio:.2f} '
                f'>={LINE_EXTRACT_STEEP_RATIO_MAX:.2f}, '
                f'steep_q95={steep_q95:.2f} '
                f'>={LINE_EXTRACT_STEEP_GRAD_Q95:.2f}'
            ),
        }

    # 边较长且整段拟合残差明显偏大：判为“带拐角/抖动边”，拒绝。
    if (len(seg_idx) >= LINE_EXTRACT_LONG_EDGE_MIN_POINTS
            and seg_mean_dist >= LINE_EXTRACT_CURVE_MEAN_DIST_MAX
            and seg_p90_dist >= LINE_EXTRACT_CURVE_P90_DIST_MAX):
        return None, {
            'segment_indices': np.array(seg_idx, dtype=np.int32),
            'fit_indices': np.array(fit_idx, dtype=np.int32),
            'num_segment_points': int(len(seg_idx)),
            'num_fit_points': int(len(fit_idx)),
            'steep_ratio': float(steep_ratio),
            'steep_q95': float(steep_q95),
            'segment_mean_dist': float(seg_mean_dist),
            'segment_p90_dist': float(seg_p90_dist),
            'reject_reason': (
                f'long edge non-line: mean_dist={seg_mean_dist:.2f} '
                f'>={LINE_EXTRACT_CURVE_MEAN_DIST_MAX:.2f}, '
                f'p90_dist={seg_p90_dist:.2f} '
                f'>={LINE_EXTRACT_CURVE_P90_DIST_MAX:.2f}'
            ),
        }

    q = float(np.clip(grad_keep_quantile, 0.35, 0.95))
    thr = float(np.quantile(grad, q)) if len(grad) > 0 else np.inf
    keep_mask = grad <= thr
    min_keep = max(2, int(round(len(fit_idx) * float(min_keep_ratio))))
    min_keep = min(min_keep, len(fit_idx))
    if int(np.sum(keep_mask)) >= min_keep:
        fit_idx = fit_idx[keep_mask]
        fit_pts = contour_pts[fit_idx]

    # 回退：筛点后不足 2 点则用 trim 后全集；仍不足则用整段轮廓
    if len(fit_pts) < 2:
        if trim * 2 < nseg - 1:
            inner = seg_idx[trim:len(seg_idx) - trim]
            fit_idx = np.array(inner, dtype=np.int32)
        else:
            fit_idx = np.array(seg_idx, dtype=np.int32)
        fit_pts = contour_pts[fit_idx]
    if len(fit_pts) < 2:
        fit_idx = np.array(seg_idx, dtype=np.int32)
        fit_pts = contour_pts[fit_idx]

    if len(fit_pts) < 2:
        return None, {}

    fit_pts_cv = fit_pts.reshape(-1, 1, 2).astype(np.float32)
    vx, vy, _x0, _y0 = cv2.fitLine(
        fit_pts_cv, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    # 用轮廓访问顺序定义有向角，避免 180/-180 歧义影响链码
    p0 = contour_pts[int(fit_idx[0])]
    p1 = contour_pts[int(fit_idx[-1])]
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    if abs(dx) + abs(dy) < 1e-6:
        dx = float(vx)
        dy = float(vy)
    # 图像坐标 y 向下，转为几何坐标（y 向上）后再算角
    angle = float(np.rad2deg(np.arctan2(-dy, dx)))
    center_x = float(np.mean(pts[:, 0]))
    center_y = float(np.mean(pts[:, 1]))

    line = Line(
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        start_angle=angle,
        end_angle=angle,
        avg_angle=angle,
        center_x=center_x,
        center_y=center_y,
        length=int(len(seg_idx)),
    )
    detail = {
        'segment_indices': np.array(seg_idx, dtype=np.int32),
        'fit_indices': np.array(fit_idx, dtype=np.int32),
        'segment_grad_threshold': float(thr),
        'segment_grad_quantile': float(q),
        'num_segment_points': int(len(seg_idx)),
        'num_fit_points': int(len(fit_idx)),
        'num_rejected_points': int(len(seg_idx) - len(fit_idx)),
        'steep_ratio': float(steep_ratio),
        'steep_q95': float(steep_q95),
        'segment_mean_dist': float(seg_mean_dist),
        'segment_p90_dist': float(seg_p90_dist),
    }
    return line, detail


def normalize_chaincode_cycle(chain: List[int]) -> List[int]:
    """循环归一化：返回字典序最小旋转（旋转不变）。"""
    if not chain:
        return []
    variants = [chain[k:] + chain[:k] for k in range(len(chain))]
    return min(variants)


def extract_line_segments_from_6_corners(
        contour: np.ndarray,
        gradient_smooth_window: int = 5,
        grad_keep_quantile: float = 0.70,
        min_keep_ratio: float = 0.45) -> Dict:
    """
    角点驱动提线：
    1) approxPolyDP 得到 6 角点
    2) 将角点映射到轮廓索引并按轮廓顺序排序
    3) 相邻角点之间每一段分别拟合一条直线，共 6 条边
    """
    if contour is None or len(contour) < 10:
        return {'ok': False, 'reason': 'contour too small'}

    approx = approx_poly_dp_n_vertices(contour, n_vertices=6)
    if approx is None or len(approx) != 6:
        return {'ok': False, 'reason': 'failed to get 6 corners'}

    contour_pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(contour_pts)
    approx_pts = approx.reshape(-1, 2).astype(np.float64)
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return {'ok': False, 'reason': 'invalid contour moment'}
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    angles = compute_contour_angles(contour, (cx, cy))
    grad = compute_angle_gradient(angles).flatten()
    grad_metric = np.abs(grad)
    if gradient_smooth_window > 1:
        grad_metric = _circular_moving_average(grad_metric, gradient_smooth_window)

    corner_indices = []
    for p in approx_pts:
        d = np.linalg.norm(contour_pts - p, axis=1)
        corner_indices.append(int(np.argmin(d)))

    uniq = sorted(set(corner_indices))
    if len(uniq) != 6:
        return {'ok': False, 'reason': f'corner index collapsed: {len(uniq)}'}

    lines: List[Line] = []
    segment_debug: List[Dict] = []
    for i in range(6):
        s = uniq[i]
        e = uniq[(i + 1) % 6]
        seg_idx = _segment_indices(s, e, n)
        line, detail = _fit_line_for_segment(
            contour_pts=contour_pts,
            contour_grad_metric=grad_metric,
            seg_idx=seg_idx,
            start_idx=s,
            end_idx=e,
            trim_ratio=0.06,
            grad_keep_quantile=grad_keep_quantile,
            min_keep_ratio=min_keep_ratio,
        )
        if line is None:
            reason = detail.get('reject_reason', 'fit failed')
            return {'ok': False, 'reason': f'edge {i + 1}: {reason}'}
        lines.append(line)
        detail['edge_id'] = int(i + 1)
        detail['edge_angle_deg'] = float(line.avg_angle)
        segment_debug.append(detail)

    ordered_corners = contour_pts[np.array(uniq, dtype=np.int32)]
    chain8 = [line_angle_to_chaincode8(line.avg_angle) for line in lines]
    lengths8 = [line.length for line in lines]
    enclosing_triangle = None
    try:
        tri_area, tri = cv2.minEnclosingTriangle(contour.astype(np.float32))
        if tri is not None and tri_area > 1.0:
            enclosing_triangle = tri.reshape(3, 2).astype(np.float64)
    except cv2.error:
        enclosing_triangle = None

    return {
        'ok': True,
        'reason': '',
        'corners': ordered_corners,
        'corner_indices': uniq,
        'lines': lines,
        'chain8': chain8,
        'chain8_normalized': normalize_chaincode_cycle(chain8),
        'lengths8': lengths8,
        'enclosing_triangle': enclosing_triangle,
        'approx': approx.reshape(-1, 2).astype(np.int32),
        'grad_keep_quantile': float(grad_keep_quantile),
        'min_keep_ratio': float(min_keep_ratio),
        'angles_flat': angles.flatten(),
        'gradient_flat': grad.flatten(),
        'gradient_metric': grad_metric,
        'segment_debug': segment_debug,
    }


def get_proc_contour_after_warp(
        image: np.ndarray,
        contour: np.ndarray) -> Optional[np.ndarray]:
    """
    与主流程一致：原图候选 → 透视矫正 → warped 红掩膜 → 取最大轮廓。
    用于总览图绿/橙判定，避免在原图透视畸变轮廓上误报失败。
    """
    warped_image, _ = warp_to_frontal(image, contour)
    if warped_image is None:
        return None
    warped_mask = extract_red_mask_from_gray_contours(warped_image)
    warped_contours = find_contours(warped_mask)
    warped_candidates = filter_contours_by_area(
        warped_contours,
        min_area=50,
        max_area=warped_image.shape[0] * warped_image.shape[1],
    )
    if not warped_candidates:
        return None
    return max(warped_candidates, key=cv2.contourArea).astype(np.int32)


def get_proc_contour_for_line_extraction(
        image: np.ndarray,
        contour: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """
    优先 warped 最大红轮廓；若 warp 后无红掩膜轮廓且尺度符合恢复窗，则退回原图轮廓。
    返回 (proc_contour, 'warped'|'original')；不可用时 (None, '')。
    """
    proc = get_proc_contour_after_warp(image, contour)
    if proc is not None:
        return proc, 'warped'
    a = float(cv2.contourArea(contour))
    if (a >= LINE_EXTRACT_RECOVER_MIN_AREA
            and a <= LINE_EXTRACT_CANDIDATE_MAX_AREA
            and len(contour) >= LINE_EXTRACT_RECOVER_MIN_POINTS):
        return contour.astype(np.int32), 'original'
    return None, ''


def _unit_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return (v / n).astype(np.float64)


def _line_dir_from_angle_deg_img(angle_deg: float) -> np.ndarray:
    """将 y 向上角度转成图像坐标系方向向量（x 右、y 下）。"""
    rad = float(np.deg2rad(float(angle_deg)))
    return _unit_vec(np.array([np.cos(rad), -np.sin(rad)], dtype=np.float64))


def _collect_backprojected_ray_dirs(
        lines: List[Line],
        inv_m: np.ndarray,
        step_px: float = 40.0,
        dedup_cos: float = 0.985) -> List[np.ndarray]:
    """
    将 warped 空间线段中心+方向逆透视到原图，得到无向射线方向集合。
    """
    dirs: List[np.ndarray] = []
    for line in lines:
        c = np.array([line.center_x, line.center_y], dtype=np.float64)
        dw = _line_dir_from_angle_deg_img(line.avg_angle)
        p0w = (c - dw * float(step_px)).astype(np.float32).reshape(1, 1, 2)
        p1w = (c + dw * float(step_px)).astype(np.float32).reshape(1, 1, 2)
        p0 = cv2.perspectiveTransform(p0w, inv_m).reshape(2).astype(np.float64)
        p1 = cv2.perspectiveTransform(p1w, inv_m).reshape(2).astype(np.float64)
        d = _unit_vec(p1 - p0)
        if not np.all(np.isfinite(d)):
            continue
        if any(abs(float(np.dot(d, u))) >= dedup_cos for u in dirs):
            continue
        dirs.append(d)
    return dirs


def _transform_point(pt: np.ndarray, mat: np.ndarray) -> np.ndarray:
    src = np.asarray(pt, dtype=np.float32).reshape(1, 1, 2)
    return cv2.perspectiveTransform(src, mat).reshape(2).astype(np.float64)


def _intersect_2d_lines(
        p1: np.ndarray,
        d1: np.ndarray,
        p2: np.ndarray,
        d2: np.ndarray) -> Optional[np.ndarray]:
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float64)
    if abs(float(np.linalg.det(A))) < 1e-9:
        return None
    t_s = np.linalg.solve(A, (p2 - p1).astype(np.float64))
    return p1 + float(t_s[0]) * d1


def _rotations(seq: List[int]) -> List[List[int]]:
    return [seq[k:] + seq[:k] for k in range(len(seq))]


def _semantic_outer_corner_from_template(
        template_variant: Optional[List[int]],
        template: Optional[List[int]] = None) -> Optional[int]:
    """
    由 weighted_chaincode_match 的最佳模板变体反推 L 外肘顶点。

    corner i 表示 edge(i-1) 与 edge(i) 的交点。模板的语义外肘在正向模板
    为 corner5；反向模板为 corner1。循环位移后按同样位移映射回当前 6 段。
    """
    if template is None:
        template = [0, 6, 4, 6, 4, 2]
    if template_variant is None or len(template_variant) != len(template):
        return None

    variant = [int(v) for v in template_variant]
    n = len(template)
    reversed_template = [((d + 4) % 8) for d in template[::-1]]
    for r, var in enumerate(_rotations(template)):
        if variant == var:
            return int((5 - r) % n)
    for r, var in enumerate(_rotations(reversed_template)):
        if variant == var:
            return int((1 - r) % n)
    return None


def _corner_geometry_for_outer_k(
        corners: np.ndarray,
        lengths: np.ndarray,
        signed_area: float,
        idx: int,
        min_angle_deg: float = 45.0,
        max_angle_deg: float = 145.0) -> Optional[Tuple[float, float, float]]:
    """返回候选顶点的 (angle, L_prev, L_next)，若不是可用外凸角则 None。"""
    n = len(corners)
    prev_pt = corners[(idx - 1) % n]
    cur_pt = corners[idx]
    next_pt = corners[(idx + 1) % n]
    incoming = cur_pt - prev_pt
    outgoing = next_pt - cur_pt
    cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
    if cross * signed_area <= 0.0:
        return None

    v_prev = _unit_vec(prev_pt - cur_pt)
    v_next = _unit_vec(next_pt - cur_pt)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(v_prev, v_next), -1.0, 1.0))))
    if angle < min_angle_deg or angle > max_angle_deg:
        return None
    return angle, float(lengths[(idx - 1) % n]), float(lengths[idx])


def _outer_hint_from_enclosing_triangle(
        corner_driven: Dict,
        signed_area: float,
        max_len: float) -> Optional[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """
    用最小外接三角形确定单个 L 的外肘与两条射线方向。

    对 L 灯条，外肘通常是外接三角形中最贴近六角轮廓顶点的三角形顶点；
    另外两个三角形顶点给出两臂外轮廓射线方向。若该映射不是可靠外凸角则返回 None。
    """
    tri_raw = corner_driven.get('enclosing_triangle')
    if tri_raw is None:
        return None
    tri = np.asarray(tri_raw, dtype=np.float64)
    corners = np.asarray(corner_driven.get('corners'), dtype=np.float64)
    lengths = np.asarray(corner_driven.get('lengths8'), dtype=np.float64)
    if tri.shape != (3, 2) or corners.shape != (6, 2) or lengths.shape[0] != 6:
        return None

    best: Optional[Tuple[float, int, int]] = None
    for tri_idx, p in enumerate(tri):
        d = np.linalg.norm(corners - p, axis=1)
        corner_idx = int(np.argmin(d))
        dist = float(d[corner_idx])
        if best is None or dist < best[0]:
            best = (dist, tri_idx, corner_idx)
    if best is None:
        return None

    dist, tri_idx, corner_idx = best
    if dist > max(28.0, 0.24 * max_len):
        return None
    geom = _corner_geometry_for_outer_k(corners, lengths, signed_area, corner_idx)
    if geom is None:
        return None
    _angle, L1, L2 = geom
    if min(L1, L2) < 0.12 * max_len:
        return None

    other = [i for i in range(3) if i != tri_idx]
    return corner_idx, tri[tri_idx], tri[other[0]], tri[other[1]]


def _choose_outer_corner_longest_adjacent_edges(corner_driven: Dict) -> Optional[int]:
    """
    六角点提线结果上选外肘：在**外凸**且夹角可用的顶点中，取**两邻边轮廓长度
    （lengths8，即该边点列长度）之和最大**的顶点。

    对应 L 的两条主臂边在六边形上共顶点，其邻边长度和通常显著大于短封口边一端。
    """
    corners = np.asarray(corner_driven.get('corners'), dtype=np.float64)
    lengths = np.asarray(corner_driven.get('lengths8'), dtype=np.float64)
    if corners.shape != (6, 2) or lengths.shape[0] != 6:
        return None
    signed_area = 0.5 * float(np.sum(
        corners[:, 0] * np.roll(corners[:, 1], -1)
        - np.roll(corners[:, 0], -1) * corners[:, 1]))
    best: Optional[Tuple[float, float, int]] = None
    for i in range(6):
        geom_info = _corner_geometry_for_outer_k(corners, lengths, signed_area, i)
        if geom_info is None:
            continue
        angle, L1, L2 = geom_info
        edge_sum = float(L1 + L2)
        ang_pen = abs(angle - 90.0)
        if best is None:
            best = (edge_sum, ang_pen, i)
            continue
        if edge_sum > best[0] + 1e-6:
            best = (edge_sum, ang_pen, i)
        elif abs(edge_sum - best[0]) <= 1e-6 and ang_pen < best[1] - 1e-6:
            best = (edge_sum, ang_pen, i)
    return None if best is None else int(best[2])


def two_longest_adjacent_outer_corner_indices(corner_driven: Dict) -> List[int]:
    """
    两个外角点（各为一对邻边拟合线的交点）：

    1) **k0**：与 `_choose_outer_corner_longest_adjacent_edges` 相同，在可外凸角上取
       **邻边 lengths8 之和最大** 的顶点（第一次筛掉的两条邻边共用的角点）。
    2) **k1**：六角闭合顺序上与 k0 **对径** 的顶点，**k1 = (k0 + 3) % 6**（与 k0±3 等价）。

    返回 [k0, k1]；若无法得到 k0 则返回 []。
    """
    corners = np.asarray(corner_driven.get('corners'), dtype=np.float64)
    lengths = np.asarray(corner_driven.get('lengths8'), dtype=np.float64)
    if corners.shape != (6, 2) or lengths.shape[0] != 6:
        return []
    signed_area = 0.5 * float(np.sum(
        corners[:, 0] * np.roll(corners[:, 1], -1)
        - np.roll(corners[:, 0], -1) * corners[:, 1]))
    scored: List[Tuple[float, float, int]] = []
    for i in range(6):
        geom_info = _corner_geometry_for_outer_k(corners, lengths, signed_area, i)
        if geom_info is None:
            continue
        angle, L1, L2 = geom_info
        edge_sum = float(L1 + L2)
        ang_pen = abs(angle - 90.0)
        scored.append((edge_sum, ang_pen, i))
    if not scored:
        return []
    scored.sort(key=lambda t: (-t[0], t[1]))
    k0 = int(scored[0][2])
    k1 = (k0 + 3) % 6
    return [k0, k1]


def draw_two_longest_adjacent_edge_pairs_on_image(
        image: np.ndarray,
        corner_driven: Dict,
        proc_src: str,
        warp_m: Optional[np.ndarray],
        line_half_len_proc: float = 240.0) -> None:
    """
    仅绘制：k0 邻边对交点 + 对径角点 k1=(k0+3)%6 的邻边对交点（两个点），
    规则见 `two_longest_adjacent_outer_corner_indices`。
    corner_driven / lines 在 proc 空间；原图就地图时需 proc_src + warp_m 做逆透视。
    """
    if not corner_driven.get('ok', False):
        return
    lines = corner_driven.get('lines')
    if not lines or len(lines) != 6:
        return
    ks = two_longest_adjacent_outer_corner_indices(corner_driven)
    if not ks:
        return
    corners = np.asarray(corner_driven['corners'], dtype=np.float64)
    inv_m: Optional[np.ndarray] = None
    if proc_src == 'warped' and warp_m is not None:
        inv_m = np.linalg.inv(warp_m).astype(np.float32)

    def proc_to_img(p: np.ndarray) -> np.ndarray:
        q = np.asarray(p, dtype=np.float64).reshape(2)
        if inv_m is not None:
            return _transform_point(q, inv_m)
        return q

    line_colors = (
        ((0, 200, 255), (0, 120, 255)),
        ((255, 140, 0), (255, 220, 100)),
    )
    pt_colors = ((0, 255, 70), (255, 0, 220))

    for pi, k_idx in enumerate(ks[:2]):
        lc0, lc1 = line_colors[pi]
        prev_line = lines[(k_idx - 1) % 6]
        next_line = lines[k_idx]
        for line, col in ((prev_line, lc0), (next_line, lc1)):
            p = np.array([line.center_x, line.center_y], dtype=np.float64)
            d = _line_dir_from_angle_deg_img(line.avg_angle)
            a = proc_to_img(p - d * float(line_half_len_proc))
            b = proc_to_img(p + d * float(line_half_len_proc))
            cv2.line(
                image,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                col, 2, cv2.LINE_AA,
            )
        p_prev = np.array([prev_line.center_x, prev_line.center_y], dtype=np.float64)
        d_prev = _line_dir_from_angle_deg_img(prev_line.avg_angle)
        p_next = np.array([next_line.center_x, next_line.center_y], dtype=np.float64)
        d_next = _line_dir_from_angle_deg_img(next_line.avg_angle)
        k_fit = _intersect_2d_lines(p_prev, d_prev, p_next, d_next)
        if k_fit is None:
            k_fit = corners[k_idx]
        k_img = proc_to_img(k_fit)
        ic = (int(round(k_img[0])), int(round(k_img[1])))
        cv2.circle(image, ic, 8, pt_colors[pi], -1, cv2.LINE_AA)
        cv2.circle(image, ic, 10, (255, 255, 255), 1, cv2.LINE_AA)


def draw_remaining_hex_edges_colored_on_image(
        image: np.ndarray,
        proc_contour: np.ndarray,
        corner_driven: Dict,
        proc_src: str,
        warp_m: Optional[np.ndarray],
        *,
        dot_radius: int = 6,
        second_pair_edge_thickness: int = 2) -> None:
    """
    绘制两个外角点（原图、蓝色实心圆）：

    1) **k0**：第一长邻边对交点（`_refined_corner_proc_at_vertex`）；
    2) **k1**：六角上对径角点 **(k0+3)%6** 的邻边对交点。

    另外画出 **k1** 处两条邻边对应的 **proc 轮廓折线**（边 (k1-1)%6 与 k1，青/橙）。
    不画 k0 处两条边轮廓、不画六角顶点。
    """
    if not corner_driven.get('ok', False):
        return
    lines = corner_driven.get('lines')
    if not lines or len(lines) != 6:
        return
    ks = two_longest_adjacent_outer_corner_indices(corner_driven)
    if not ks:
        return

    inv_m: Optional[np.ndarray] = None
    if proc_src == 'warped' and warp_m is not None:
        inv_m = np.linalg.inv(warp_m).astype(np.float32)

    def proc_to_img(p: np.ndarray) -> np.ndarray:
        q = np.asarray(p, dtype=np.float64).reshape(2)
        if inv_m is not None:
            return _transform_point(q, inv_m)
        return q

    contour_pts = proc_contour.reshape(-1, 2).astype(np.float64)
    n = int(contour_pts.shape[0])
    if len(ks) > 1 and n >= 3:
        k1 = int(ks[1])
        pair_edges = ((k1 - 1) % 6, k1)
        edge_colors = ((255, 255, 0), (0, 165, 255))
        th = max(1, int(second_pair_edge_thickness))
        for j, e in enumerate(pair_edges):
            line = lines[e]
            seg_idx = _segment_indices(line.start_idx, line.end_idx, n)
            if len(seg_idx) < 2:
                continue
            strip = contour_pts[seg_idx]
            img_row = []
            for p in strip:
                q = proc_to_img(p)
                img_row.append([int(round(q[0])), int(round(q[1]))])
            arr = np.asarray(img_row, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                image, [arr], isClosed=False, color=edge_colors[j % 2],
                thickness=th, lineType=cv2.LINE_AA)

    blue_bgr = (255, 0, 0)
    r = max(3, int(dot_radius))
    for k_idx in ks[:2]:
        k_proc = _refined_corner_proc_at_vertex(corner_driven, int(k_idx))
        q = proc_to_img(k_proc)
        c = (int(round(q[0])), int(round(q[1])))
        cv2.circle(image, c, r, blue_bgr, -1, cv2.LINE_AA)


def _choose_outer_corner_from_six(
        corner_driven: Dict,
        template_variant: Optional[List[int]] = None,
        match_score: float = -1.0) -> Tuple[Optional[int], str]:
    """
    在 6 角点闭合多边形中选外凸拐点（L 的两臂交汇处）。

    优先级：
    1) 邻边 lengths8 之和最大的外凸顶点（与六角点提线同源）；
    2) 最小外接三角形（回退）；
    3) 局部几何打分 + 链码模板语义覆盖（再回退）。

    Returns:
        (k_idx 或 None, 来源标签 longest_adjacent | enclosing_triangle | template_semantic | local_geometry)
    """
    corners = np.asarray(corner_driven.get('corners'), dtype=np.float64)
    lengths = np.asarray(corner_driven.get('lengths8'), dtype=np.float64)
    if corners.shape != (6, 2) or lengths.shape[0] != 6:
        return None, 'none'

    signed_area = 0.5 * float(np.sum(
        corners[:, 0] * np.roll(corners[:, 1], -1)
        - np.roll(corners[:, 0], -1) * corners[:, 1]))
    max_len = max(float(np.max(lengths)), 1.0)

    k_long = _choose_outer_corner_longest_adjacent_edges(corner_driven)
    if k_long is not None:
        return k_long, 'longest_adjacent'

    tri_hint = _outer_hint_from_enclosing_triangle(corner_driven, signed_area, max_len)
    if tri_hint is not None:
        return int(tri_hint[0]), 'enclosing_triangle'

    # 相对阈值：短边端点处 min(邻边) 往往远小于最长边
    short_stub_ratio = 0.11
    stub_penalty = 0.42
    semantic_override_margin = 0.18
    semantic_idx = _semantic_outer_corner_from_template(template_variant)
    semantic_score: Optional[float] = None
    semantic_min_adj_len = 0.0
    best: Optional[Tuple[float, int]] = None

    for i in range(6):
        geom_info = _corner_geometry_for_outer_k(corners, lengths, signed_area, i)
        if geom_info is None:
            continue
        angle, L1, L2 = geom_info
        geom = float(np.sqrt(max(L1 * L2, 0.0))) / max_len
        angle_score = 1.0 - min(abs(angle - 90.0), 90.0) / 90.0
        balance = min(L1, L2) / (max(L1, L2) + 1e-9)
        score = geom + 0.55 * angle_score + 0.18 * balance
        if min(L1, L2) < short_stub_ratio * max_len:
            score -= stub_penalty
        if i == semantic_idx:
            semantic_score = score
            semantic_min_adj_len = min(L1, L2)
        if best is None or score > best[0]:
            best = (score, i)

    if (best is not None
            and semantic_idx is not None
            and semantic_score is not None
            and match_score >= 0.58
            and semantic_min_adj_len >= 0.15 * max_len
            and semantic_score >= best[0] - semantic_override_margin):
        return semantic_idx, 'template_semantic'

    if best is None:
        return None, 'none'
    return int(best[1]), 'local_geometry'


def _refined_corner_proc_at_vertex(corner_driven: Dict, k_idx: int) -> np.ndarray:
    """顶点 k 处两邻边拟合线交点；失败或偏离过大则退回六角角点。"""
    corners = np.asarray(corner_driven['corners'], dtype=np.float64)
    lines = corner_driven.get('lines', [])
    k_proc = corners[k_idx].astype(np.float64).copy()
    if len(lines) != 6:
        return k_proc
    prev_line = lines[(k_idx - 1) % 6]
    next_line = lines[k_idx]
    p_prev = np.array([prev_line.center_x, prev_line.center_y], dtype=np.float64)
    p_next = np.array([next_line.center_x, next_line.center_y], dtype=np.float64)
    d_prev = _line_dir_from_angle_deg_img(prev_line.avg_angle)
    d_next = _line_dir_from_angle_deg_img(next_line.avg_angle)
    k_fit = _intersect_2d_lines(p_prev, d_prev, p_next, d_next)
    if k_fit is not None and float(np.linalg.norm(k_fit - k_proc)) <= 45.0:
        return k_fit
    return k_proc


def build_lbar_observation_from_six_corners(
        candidate_idx: int,
        contour: np.ndarray,
        corner_driven: Dict,
        proc_src: str,
        warp_m: Optional[np.ndarray],
        match_score: float = -1.0,
        template_variant: Optional[List[int]] = None) -> Optional["lba.LBarObservation"]:
    """
    六角提线成功后：仅用「最长邻边对」与「去掉该对边后余边上最长邻边对」两个外角点；
    主观测 corner_k 取第一角点；corner_k2 取第二角点。不再做链码打分与 _choose_outer_corner 回退。
    match_score / template_variant 参数保留兼容，已忽略。
    """
    from calculation import l_bar_association as lba

    _ = match_score, template_variant
    ks = two_longest_adjacent_outer_corner_indices(corner_driven)
    if not ks:
        return None
    k_idx = int(ks[0])
    lines = corner_driven.get('lines', [])
    if len(lines) != 6:
        return None

    k_proc = _refined_corner_proc_at_vertex(corner_driven, k_idx)
    prev_line = lines[(k_idx - 1) % 6]
    next_line = lines[k_idx]
    p_prev = np.array([prev_line.center_x, prev_line.center_y], dtype=np.float64)
    p_next = np.array([next_line.center_x, next_line.center_y], dtype=np.float64)
    prev_dir_proc = _unit_vec(p_prev - k_proc)
    next_dir_proc = _unit_vec(p_next - k_proc)

    prev_sample = k_proc + prev_dir_proc * 70.0
    next_sample = k_proc + next_dir_proc * 70.0

    corner_k2_img: Optional[np.ndarray] = None
    if len(ks) > 1:
        k1_proc = _refined_corner_proc_at_vertex(corner_driven, int(ks[1]))
        if proc_src == 'warped':
            if warp_m is not None:
                inv_m = np.linalg.inv(warp_m).astype(np.float32)
                corner_k2_img = _transform_point(k1_proc, inv_m)
        else:
            corner_k2_img = k1_proc.astype(np.float64)

    if proc_src == 'warped':
        if warp_m is None:
            return None
        inv_m = np.linalg.inv(warp_m).astype(np.float32)
        k = _transform_point(k_proc, inv_m)
        p_prev = _transform_point(prev_sample, inv_m)
        p_next = _transform_point(next_sample, inv_m)
    else:
        k = k_proc.astype(np.float64)
        p_prev = prev_sample.astype(np.float64)
        p_next = next_sample.astype(np.float64)

    dir_a = _unit_vec(p_prev - k)
    dir_b = _unit_vec(p_next - k)
    bisector = _unit_vec(dir_a + dir_b)

    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    centroid = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']], dtype=np.float64)
    if float(np.dot(bisector, centroid - k)) < 0.0:
        bisector = -bisector

    return lba.LBarObservation(
        candidate_idx=candidate_idx,
        centroid=centroid,
        corner_k=k,
        dir_a=dir_a,
        dir_b=dir_b,
        bisector=bisector,
        area=float(cv2.contourArea(contour)),
        match_score=-1.0,
        arm_a_end=p_prev,
        arm_b_end=p_next,
        corner_k2=corner_k2_img,
    )


def _build_ray_hint_from_dirs(
        ray_dirs: List[np.ndarray],
        ref_a: np.ndarray,
        ref_b: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    在逆透视射线中选择与参考两臂最一致的一对，并给出角中位线方向。
    """
    if len(ray_dirs) < 2:
        return None
    ra = _unit_vec(ref_a)
    rb = _unit_vec(ref_b)
    best: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
    for i in range(len(ray_dirs)):
        for j in range(i + 1, len(ray_dirs)):
            d1 = _unit_vec(ray_dirs[i])
            d2 = _unit_vec(ray_dirs[j])
            s11 = abs(float(np.dot(d1, ra))) + abs(float(np.dot(d2, rb)))
            s12 = abs(float(np.dot(d1, rb))) + abs(float(np.dot(d2, ra)))
            if s12 > s11:
                d1, d2 = d2, d1
                align = s12
            else:
                align = s11
            ortho = 1.0 - abs(float(np.dot(d1, d2)))
            score = align + 0.8 * ortho
            if best is None or score > best[0]:
                best = (score, d1, d2)
    if best is None:
        return None
    _, h1, h2 = best
    if abs(float(np.dot(h1, h2))) > 0.90:
        return None
    if float(np.dot(h1, ra)) < 0:
        h1 = -h1
    if float(np.dot(h2, rb)) < 0:
        h2 = -h2
    hb = _unit_vec(h1 + h2)
    return h1, h2, hb


def _passes_ray_hint_consistency(
        hint: Tuple[np.ndarray, np.ndarray, np.ndarray],
        ref_a: np.ndarray,
        ref_b: np.ndarray,
        ref_bisector: np.ndarray,
        arm_cos_min: float = 0.55,
        bisector_cos_min: float = 0.35) -> bool:
    """射线-角点一致性筛选：两臂和中位线都不能偏离太多。"""
    ha, hb, hm = hint
    ra = _unit_vec(ref_a)
    rb = _unit_vec(ref_b)
    rbis = _unit_vec(ref_bisector)
    a11 = abs(float(np.dot(ha, ra)))
    a12 = abs(float(np.dot(ha, rb)))
    a21 = abs(float(np.dot(hb, ra)))
    a22 = abs(float(np.dot(hb, rb)))
    arm_fit = max(min(a11, a22), min(a12, a21))
    bis_fit = abs(float(np.dot(hm, rbis)))
    return arm_fit >= arm_cos_min and bis_fit >= bisector_cos_min


def _blend_observation_with_ray_hint(
        dir_a: np.ndarray,
        dir_b: np.ndarray,
        bisector: np.ndarray,
        hint: Tuple[np.ndarray, np.ndarray, np.ndarray],
        alpha: float = 0.72) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    用逆透视射线对原始 keypoint 方向做平滑修正，避免完全替换导致抖动。
    """
    ha, hb, hm = hint
    ra = _unit_vec(dir_a)
    rb = _unit_vec(dir_b)
    if abs(float(np.dot(ha, rb))) + abs(float(np.dot(hb, ra))) > abs(float(np.dot(ha, ra))) + abs(float(np.dot(hb, rb))):
        ha, hb = hb, ha
    if float(np.dot(ha, ra)) < 0:
        ha = -ha
    if float(np.dot(hb, rb)) < 0:
        hb = -hb
    rbis = _unit_vec(bisector)
    if float(np.dot(hm, rbis)) < 0:
        hm = -hm
    na = _unit_vec((1.0 - alpha) * ra + alpha * ha)
    nb = _unit_vec((1.0 - alpha) * rb + alpha * hb)
    nm = _unit_vec((1.0 - alpha) * rbis + alpha * hm)
    return na, nb, nm


def _remap_line_to_original(
        line: Line,
        roll_k: int,
        n: int,
        orig_pts: np.ndarray,
        angles_orig_flat: np.ndarray) -> Line:
    """将 rolled 空间中的线段映射回原始轮廓下标；rolled[j] 对应 orig[(j + roll_k) % n]。"""
    if roll_k == 0:
        return line
    s = (line.start_idx + roll_k) % n
    e = (line.end_idx + roll_k) % n
    if s <= e:
        span = orig_pts[s : e + 1]
        ang_slice = angles_orig_flat[s : e + 1]
    else:
        span = np.vstack([orig_pts[s:], orig_pts[: e + 1]])
        ang_slice = np.concatenate([angles_orig_flat[s:], angles_orig_flat[: e + 1]])
    return Line(
        start_idx=int(s),
        end_idx=int(e),
        start_angle=float(angles_orig_flat[s]),
        end_angle=float(angles_orig_flat[e]),
        avg_angle=float(np.mean(ang_slice)),
        center_x=float(np.mean(span[:, 0])),
        center_y=float(np.mean(span[:, 1])),
        length=line.length,
    )


def compute_adaptive_straight_threshold(
        gradient_flat: np.ndarray,
        fixed_fallback: float,
        k_peaks: int = 6,
        alpha: float = 0.45,
        min_peak_distance: int = 5) -> float:
    """
    阶段 A：在 |梯度| 的闭合序列上找局部极大，NMS 取前 k_peaks 个峰；
    τ = 这 k 个峰高的最小值（最弱的角）；直线带阈值为 α·τ。
    峰不足或非有限值时回退 fixed_fallback。
    """
    g = np.abs(np.asarray(gradient_flat, dtype=np.float64)).ravel()
    n = len(g)
    if n < 5 or k_peaks < 1:
        return float(fixed_fallback)

    peaks: List[Tuple[int, float]] = []
    for i in range(n):
        if g[i] > g[(i - 1) % n] and g[i] > g[(i + 1) % n]:
            peaks.append((i, float(g[i])))

    if len(peaks) < k_peaks:
        return float(fixed_fallback)

    peaks.sort(key=lambda t: -t[1])
    selected: List[Tuple[int, float]] = []
    for idx, val in peaks:
        if all(_wrap_index_dist(idx, sj, n) >= min_peak_distance for sj, _ in selected):
            selected.append((idx, val))
            if len(selected) >= k_peaks:
                break

    if len(selected) < k_peaks:
        return float(fixed_fallback)

    tau = min(val for _, val in selected)
    thr = alpha * tau
    if not np.isfinite(thr) or thr <= 0:
        return float(fixed_fallback)
    return float(thr)


def extract_line_segments(
        contour: np.ndarray,
        gradient_threshold: float = 3.0,
        min_length: int = 2,
        top_k: int = 6,
        merge_max_gap: int = 20,
        merge_max_angle_diff: float = 12.0,
        use_adaptive_gradient: bool = True,
        adaptive_k_peaks: int = 6,
        adaptive_alpha: float = 0.45,
        min_peak_distance: int = 5,
        gradient_smooth_window: int = 5,
        bridge_max_gap: int = 2,
        roll_seam_to_corner: bool = False) -> Tuple[np.ndarray, np.ndarray, List[Line], float]:
    """
    提取轮廓的直线段

    基于角度曲线 + 梯度：
    0. 可选：在原始轮廓上粗算梯度，将轮廓 np.roll 使最强角点落在下标 0，再提线（返回的角度/梯度/线段
       下标均对齐**原始**轮廓顺序）
    1. 计算角度曲线（高斯平滑方向向量）
    2. 计算角度对弧长索引的梯度
    3. 直线判定：|梯度| ≤ straight_thr（可自适应）
    4. 按轮廓点索引顺序合并相邻且角度相近的段
    5. 按合并后长度降序，只保留最长的 top_k 条

    Args:
        roll_seam_to_corner: 是否将闭合接缝滚到最显著角点，减轻长边被下标 0 切开

    Returns:
        (angles, gradient, lines, straight_thr)，lines 下标与输入 contour 一致
    """
    if len(contour) < 10:
        return None, None, [], float(gradient_threshold)

    orig_pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(orig_pts)
    roll_k = 0
    contour_work = contour

    M0 = cv2.moments(contour)
    if M0['m00'] == 0:
        return None, None, [], float(gradient_threshold)
    cx0 = M0['m10'] / M0['m00']
    cy0 = M0['m01'] / M0['m00']

    if roll_seam_to_corner:
        angles_probe = compute_contour_angles(contour, (cx0, cy0))
        grad_probe = compute_angle_gradient(angles_probe).flatten()
        roll_k = _find_corner_roll_index(grad_probe)
        if roll_k != 0:
            rolled = np.roll(orig_pts, -roll_k, axis=0)
            contour_work = rolled.reshape(-1, 1, 2).astype(contour.dtype)

    M = cv2.moments(contour_work)
    if M['m00'] == 0:
        return None, None, [], float(gradient_threshold)
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    # 步骤1–2: 在（可能已 roll 的）轮廓上算角度与梯度
    angles = compute_contour_angles(contour_work, (cx, cy))
    gradient = compute_angle_gradient(angles)

    angles_flat = angles.flatten()
    gradient_flat = gradient.flatten()
    gradient_metric = np.abs(gradient_flat)
    if gradient_smooth_window > 1:
        gradient_metric = _circular_moving_average(gradient_metric, gradient_smooth_window)

    if use_adaptive_gradient:
        straight_thr = compute_adaptive_straight_threshold(
            gradient_metric,
            fixed_fallback=gradient_threshold,
            k_peaks=adaptive_k_peaks,
            alpha=adaptive_alpha,
            min_peak_distance=min_peak_distance,
        )
    else:
        straight_thr = float(gradient_threshold)

    # 步骤3: 提取直线段（|梯度| ≤ straight_thr 的连续区间）
    straight_mask = gradient_metric <= straight_thr
    straight_mask = _bridge_small_straight_gaps(straight_mask, bridge_max_gap)
    lines = []
    in_line = False
    start_idx = 0

    for i in range(n):
        if straight_mask[i]:
            if not in_line:
                in_line = True
                start_idx = i
        else:
            if in_line:
                in_line = False
                # 检查长度
                if i - start_idx >= min_length:
                    # 计算线段信息
                    start_angle = angles_flat[start_idx]
                    end_angle = angles_flat[i - 1]
                    avg_angle = _mean_line_orientation_deg(angles_flat[start_idx:i])

                    # 计算中心点（与 contour_work 一致）
                    pts = contour_work.reshape(-1, 2)
                    center_pts = pts[start_idx:i]
                    center_x = np.mean(center_pts[:, 0])
                    center_y = np.mean(center_pts[:, 1])

                    lines.append(Line(
                        start_idx=start_idx,
                        end_idx=i - 1,
                        start_angle=start_angle,
                        end_angle=end_angle,
                        avg_angle=avg_angle,
                        center_x=center_x,
                        center_y=center_y,
                        length=i - start_idx
                    ))

    # 处理最后一个线段
    if in_line and n - start_idx >= min_length:
        start_angle = angles_flat[start_idx]
        end_angle = angles_flat[n - 1]
        avg_angle = _mean_line_orientation_deg(angles_flat[start_idx:n])

        pts = contour_work.reshape(-1, 2)
        center_pts = pts[start_idx:n]
        center_x = np.mean(center_pts[:, 0])
        center_y = np.mean(center_pts[:, 1])

        lines.append(Line(
            start_idx=start_idx,
            end_idx=n - 1,
            start_angle=start_angle,
            end_angle=end_angle,
            avg_angle=avg_angle,
            center_x=center_x,
            center_y=center_y,
            length=n - start_idx
        ))

    # 按轮廓索引顺序合并：相邻段且平均角度相近 → 一条线（一个 Line / 一个显示编号）
    lines.sort(key=lambda L: (L.start_idx, L.end_idx))
    if len(lines) >= 2:
        merged_lines = [lines[0]]
        pts = contour_work.reshape(-1, 2)

        for i in range(1, len(lines)):
            prev_line = merged_lines[-1]
            curr_line = lines[i]

            gap = curr_line.start_idx - prev_line.end_idx - 1
            adiff = _line_orientation_diff_deg(prev_line.avg_angle, curr_line.avg_angle)

            if gap <= merge_max_gap and adiff <= merge_max_angle_diff:
                merged_start = prev_line.start_idx
                merged_end = curr_line.end_idx
                merged_avg_angle = _mean_line_orientation_deg(angles_flat[merged_start:merged_end + 1])
                center_pts = pts[merged_start:merged_end + 1]
                merged_lines[-1] = Line(
                    start_idx=merged_start,
                    end_idx=merged_end,
                    start_angle=prev_line.start_angle,
                    end_angle=curr_line.end_angle,
                    avg_angle=merged_avg_angle,
                    center_x=float(np.mean(center_pts[:, 0])),
                    center_y=float(np.mean(center_pts[:, 1])),
                    length=merged_end - merged_start + 1,
                )
            else:
                merged_lines.append(curr_line)

        lines = merged_lines

    # 闭合轮廓首尾补充合并：处理“接缝落在同一直线中间”导致的一分为二
    if len(lines) >= 2:
        first = lines[0]
        last = lines[-1]
        wrap_gap = first.start_idx + n - last.end_idx - 1
        wrap_adiff = _line_orientation_diff_deg(last.avg_angle, first.avg_angle)
        if wrap_gap <= merge_max_gap and wrap_adiff <= merge_max_angle_diff:
            s = last.start_idx
            e = first.end_idx  # 允许 s > e，表示跨接缝的环绕线段
            span_pts = np.vstack([pts[s:], pts[: e + 1]])
            ang_span = np.concatenate([angles_flat[s:], angles_flat[: e + 1]])
            merged_wrap = Line(
                start_idx=s,
                end_idx=e,
                start_angle=last.start_angle,
                end_angle=first.end_angle,
                avg_angle=_mean_line_orientation_deg(ang_span),
                center_x=float(np.mean(span_pts[:, 0])),
                center_y=float(np.mean(span_pts[:, 1])),
                length=len(ang_span),
            )
            lines = lines[1:-1] + [merged_wrap]

    # 尾/头短段补偿合并：处理“主合并后仍残留在序列端点的短噪声段”
    # 典型现象：Line5 很短且方向接近 Line4，但未在第一轮被吸收
    if len(lines) >= 2:
        pts = contour_work.reshape(-1, 2)
        ordered = sort_lines_in_contour_order(lines, n)
        short_edge_thr = max(min_length + 1, 6)

        # 1) 尾短段与其前邻段合并（非 wrap）
        if len(ordered) >= 2:
            prev_line = ordered[-2]
            tail_line = ordered[-1]
            tail_gap = (tail_line.start_idx - prev_line.end_idx - 1) % n
            tail_adiff = _line_orientation_diff_deg(prev_line.avg_angle, tail_line.avg_angle)
            if tail_line.length <= short_edge_thr and tail_gap <= merge_max_gap and tail_adiff <= merge_max_angle_diff:
                s = prev_line.start_idx
                e = tail_line.end_idx
                if s <= e:
                    span_pts = pts[s:e + 1]
                    ang_span = angles_flat[s:e + 1]
                else:
                    span_pts = np.vstack([pts[s:], pts[:e + 1]])
                    ang_span = np.concatenate([angles_flat[s:], angles_flat[:e + 1]])
                merged_tail = Line(
                    start_idx=s,
                    end_idx=e,
                    start_angle=prev_line.start_angle,
                    end_angle=tail_line.end_angle,
                    avg_angle=_mean_line_orientation_deg(ang_span),
                    center_x=float(np.mean(span_pts[:, 0])),
                    center_y=float(np.mean(span_pts[:, 1])),
                    length=len(ang_span),
                )
                ordered = ordered[:-2] + [merged_tail]

        # 2) 头短段与其后邻段合并（非 wrap）
        if len(ordered) >= 2:
            head_line = ordered[0]
            next_line = ordered[1]
            head_gap = (next_line.start_idx - head_line.end_idx - 1) % n
            head_adiff = _line_orientation_diff_deg(head_line.avg_angle, next_line.avg_angle)
            if head_line.length <= short_edge_thr and head_gap <= merge_max_gap and head_adiff <= merge_max_angle_diff:
                s = head_line.start_idx
                e = next_line.end_idx
                if s <= e:
                    span_pts = pts[s:e + 1]
                    ang_span = angles_flat[s:e + 1]
                else:
                    span_pts = np.vstack([pts[s:], pts[:e + 1]])
                    ang_span = np.concatenate([angles_flat[s:], angles_flat[:e + 1]])
                merged_head = Line(
                    start_idx=s,
                    end_idx=e,
                    start_angle=head_line.start_angle,
                    end_angle=next_line.end_angle,
                    avg_angle=_mean_line_orientation_deg(ang_span),
                    center_x=float(np.mean(span_pts[:, 0])),
                    center_y=float(np.mean(span_pts[:, 1])),
                    length=len(ang_span),
                )
                ordered = [merged_head] + ordered[2:]

        lines = ordered

    lines = [L for L in lines if L.length >= min_length]
    lines.sort(key=lambda L: L.length, reverse=True)
    if top_k > 0:
        lines = lines[:top_k]

    # 线段下标映射回原始 contour；角度/梯度在**原始**轮廓上重算，避免对 unwrap 序列 np.roll 产生断点
    if roll_k != 0:
        angles_out = compute_contour_angles(contour, (cx0, cy0))
        ao_full = angles_out.flatten()
        lines = [
            _remap_line_to_original(L, roll_k, n, orig_pts, ao_full)
            for L in lines
        ]
        gradient_out = compute_angle_gradient(angles_out)
    else:
        angles_out = angles
        gradient_out = gradient

    return angles_out, gradient_out, lines, straight_thr


def visualize_corner_driven_edges(
        image: np.ndarray,
        contour: np.ndarray,
        corners: np.ndarray,
        lines: List[Line],
        output_path: str) -> Optional[str]:
    """可视化角点驱动提取的 6 条边。"""
    if image is None or contour is None or corners is None or not lines:
        return None

    vis = image.copy()
    contour_pts = contour.reshape(-1, 2)
    cv2.polylines(vis, [contour_pts.astype(np.int32)], True, (80, 80, 80), 1)

    colors = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 220, 80), (220, 80, 255), (80, 255, 255),
    ]

    n = len(contour_pts)
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        seg_idx = _segment_indices(line.start_idx, line.end_idx, n)
        seg = contour_pts[seg_idx].astype(np.int32)
        cv2.polylines(vis, [seg], False, color, 2)
        cx, cy = int(line.center_x), int(line.center_y)
        cv2.putText(
            vis,
            f"L{i+1}:{line.avg_angle:.0f}",
            (cx + 2, cy - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

    for i, p in enumerate(corners):
        x, y = int(p[0]), int(p[1])
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
        cv2.circle(vis, (x, y), 6, (255, 255, 255), 1)
        cv2.putText(
            vis,
            f"C{i+1}",
            (x + 6, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

    ok = cv2.imwrite(output_path, vis)
    if ok:
        return output_path
    fallback_dir = "/tmp/vision_debug_line_segments"
    os.makedirs(fallback_dir, exist_ok=True)
    fallback_path = os.path.join(fallback_dir, os.path.basename(output_path))
    if cv2.imwrite(fallback_path, vis):
        return fallback_path
    return None


def visualize_corner_gradient_pipeline(
        image: np.ndarray,
        contour: np.ndarray,
        corners: np.ndarray,
        lines: List[Line],
        angles_flat: np.ndarray,
        gradient_flat: np.ndarray,
        gradient_metric: np.ndarray,
        segment_debug: List[Dict],
        chain8: List[int],
        chain8_norm: List[int],
        chain8_used: List[int],
        score: float,
        output_path: str) -> Optional[str]:
    """综合可视化：筛点+拟合线+角度图+梯度图+链码信息。"""
    if image is None or contour is None or corners is None or not lines:
        return None

    contour_pts = contour.reshape(-1, 2)
    n = len(contour_pts)
    idx = np.arange(n)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)

    # Panel1: 轮廓 + 角点 + 过滤点 + 拟合段
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.plot(contour_pts[:, 0], contour_pts[:, 1], color='lightgray', linewidth=1, alpha=0.7, label='contour')

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(lines), 1)))
    for i, line in enumerate(lines):
        c = colors[i]
        seg_idx = _segment_indices(line.start_idx, line.end_idx, n)
        seg = contour_pts[seg_idx]
        ax1.plot(seg[:, 0], seg[:, 1], '-', color=c, linewidth=2.5, label=f'E{i+1}')
        # 显式叠加“拟合直线”（由 avg_angle + segment center 定义），便于区分轮廓折线与拟合结果
        theta = np.deg2rad(float(line.avg_angle))
        line_dir = np.array([np.cos(theta), -np.sin(theta)], dtype=np.float64)
        seg_span = float(np.linalg.norm(seg[-1] - seg[0])) if len(seg) >= 2 else 0.0
        fit_half = max(16.0, 0.65 * max(seg_span, float(line.length)))
        fit_center = np.array([line.center_x, line.center_y], dtype=np.float64)
        fit_a = fit_center - fit_half * line_dir
        fit_b = fit_center + fit_half * line_dir
        ax1.plot(
            [fit_a[0], fit_b[0]],
            [fit_a[1], fit_b[1]],
            '--',
            color=c,
            linewidth=2.0,
            alpha=0.95,
        )
        ax1.text(line.center_x, line.center_y, f'E{i+1}:{line.avg_angle:.0f}°', fontsize=8, color='white',
                 bbox=dict(boxstyle='round', facecolor=c, alpha=0.75))

        if i < len(segment_debug):
            dbg = segment_debug[i]
            fit_idx = np.asarray(dbg.get('fit_indices', []), dtype=np.int32)
            if len(fit_idx) > 0:
                fit_pts = contour_pts[fit_idx]
                ax1.scatter(fit_pts[:, 0], fit_pts[:, 1], s=8, color=c, alpha=0.9)
            seg_all = np.asarray(dbg.get('segment_indices', []), dtype=np.int32)
            if len(seg_all) > 0:
                reject_idx = np.setdiff1d(seg_all, fit_idx)
                if len(reject_idx) > 0:
                    rej_pts = contour_pts[reject_idx]
                    ax1.scatter(rej_pts[:, 0], rej_pts[:, 1], s=10, color='red', alpha=0.4)

    for i, p in enumerate(corners):
        x, y = p[0], p[1]
        ax1.scatter([x], [y], s=45, color='yellow', edgecolors='black')
        ax1.text(x + 5, y - 4, f'C{i+1}', color='yellow', fontsize=9, weight='bold')
    ax1.set_title('Corner-driven fitting: kept/rejected points + fitted lines')
    ax1.axis('equal')
    ax1.grid(alpha=0.2)

    # Panel2: 角度曲线
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(idx, angles_flat, color='tab:blue', linewidth=1.2, label='angle')
    for i, line in enumerate(lines):
        c = colors[i]
        if line.start_idx <= line.end_idx:
            ax2.axvspan(line.start_idx, line.end_idx, alpha=0.15, color=c)
        else:
            ax2.axvspan(line.start_idx, n - 1, alpha=0.15, color=c)
            ax2.axvspan(0, line.end_idx, alpha=0.15, color=c)
    ax2.set_title('Angle curve (with edge spans)')
    ax2.set_xlabel('Contour index')
    ax2.set_ylabel('Angle (deg)')
    ax2.grid(alpha=0.25)

    # Panel3: 梯度曲线 + 每段阈值
    ax3 = fig.add_subplot(gs[1, 1])
    gabs = np.abs(gradient_flat)
    ax3.plot(idx, gabs, color='tab:green', linewidth=1.2, label='|gradient|')
    ax3.plot(idx, gradient_metric, color='tab:orange', linewidth=1.0, label='smoothed |gradient|')
    for i, dbg in enumerate(segment_debug):
        c = colors[i % len(colors)]
        thr = float(dbg.get('segment_grad_threshold', 0.0))
        seg_all = np.asarray(dbg.get('segment_indices', []), dtype=np.int32)
        if len(seg_all) > 0:
            ax3.scatter(seg_all, gradient_metric[seg_all], s=8, color=c, alpha=0.15)
            fit_idx = np.asarray(dbg.get('fit_indices', []), dtype=np.int32)
            if len(fit_idx) > 0:
                ax3.scatter(fit_idx, gradient_metric[fit_idx], s=10, color=c, alpha=0.75)
        ax3.axhline(thr, color=c, linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_title('Gradient filtering per edge (kept points shown)')
    ax3.set_xlabel('Contour index')
    ax3.set_ylabel('|gradient|')
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=8, loc='upper right')

    info = (
        f"chain8(raw): {chain8}\n"
        f"chain8(norm-rotation-invariant): {chain8_norm}\n"
        f"chain8(used): {chain8_used}\n"
        f"weighted score: {score:.3f}\n"
    )
    fig.text(0.53, 0.02, info, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    try:
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()
        return output_path
    except PermissionError:
        fallback_dir = "/tmp/vision_debug_line_segments"
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_path = os.path.join(fallback_dir, os.path.basename(output_path))
        plt.savefig(fallback_path, dpi=120, bbox_inches='tight')
        plt.close()
        return fallback_path


def visualize_line_segments(
        image,
        contour,
        candidate_idx,
        angles,
        gradient,
        lines,
        output_dir,
        area_ratio=None,
        gradient_threshold: float = 3.0,
        min_length: int = 2,
        top_k: int = 6,
        gradient_fallback_ref: Optional[float] = None):
    """可视化直线段提取结果（默认展示最长的 top_k 条）"""

    if angles is None or gradient is None:
        return None

    angles_flat = angles.flatten()
    gradient_flat = gradient.flatten()
    indices = np.arange(len(angles_flat))

    # 创建图形
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # 1. 原图 + 轮廓 + 直线段标注
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 绘制轮廓
    contour_pts = contour.reshape(-1, 2)
    ax1.plot(contour_pts[:, 0], contour_pts[:, 1], 'b-', linewidth=1, alpha=0.5, label='Contour')

    # 绘制直线段（已按长度从长到短排序，#1 为最长）
    n_show = len(lines)
    colors = plt.cm.rainbow(np.linspace(0, 1, max(n_show, 1)))
    for idx, line in enumerate(lines):
        if line.start_idx <= line.end_idx:
            seg_pts = contour_pts[line.start_idx:line.end_idx+1]
        else:
            seg_pts = np.vstack([contour_pts[line.start_idx:], contour_pts[:line.end_idx+1]])
        ax1.plot(seg_pts[:, 0], seg_pts[:, 1], '-', color=colors[idx], linewidth=3,
                label=f'#{idx+1} len={line.length}')

        # 标注线段编号和角度
        ax1.text(line.center_x, line.center_y, f'{idx+1}\n{line.avg_angle:.0f}°',
                color='white', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.8),
                ha='center', va='center')

    ax1.set_title(
        f'Candidate #{candidate_idx}: top {n_show} longest lines '
        f'(|grad|≤{gradient_threshold}, min_len={min_length}, cap={top_k})',
        fontsize=12)
    ax1.axis('equal')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. 直线段信息表
    ax2 = fig.add_subplot(gs[0:2, 1])
    text_info = f"Line Segments Analysis (longest {top_k} kept)\n"
    text_info += f"{'='*60}\n"
    text_info += f"Shown segments: {len(lines)}\n"
    if area_ratio is not None:
        text_info += f"Warped area ratio: {area_ratio:.4f}\n"
    text_info += f"Effective |grad| straight band: ±{gradient_threshold:.4f}\n"
    if gradient_fallback_ref is not None and abs(gradient_fallback_ref - gradient_threshold) > 1e-6:
        text_info += f"Fixed fallback reference: ±{gradient_fallback_ref:.4f}\n"
    text_info += f"Min segment length: {min_length} points\n\n"

    for idx, line in enumerate(lines):
        text_info += f"Line {idx+1}:\n"
        text_info += f"  Indices: [{line.start_idx}, {line.end_idx}]\n"
        text_info += f"  Length: {line.length} points\n"
        text_info += f"  Avg angle: {line.avg_angle:.1f}°\n"
        text_info += f"  Angle range: [{line.start_angle:.1f}°, {line.end_angle:.1f}°]\n"
        text_info += f"  Center: ({line.center_x:.0f}, {line.center_y:.0f})\n\n"

    ax2.text(0.05, 0.95, text_info, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.axis('off')

    # 3. 角度曲线 + 直线段标注
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(indices, angles_flat, 'b-', linewidth=1.5, label='Angle curve')

    # 标注直线段区间
    for idx, line in enumerate(lines):
        if line.start_idx <= line.end_idx:
            ax3.axvspan(line.start_idx, line.end_idx, alpha=0.3, color=colors[idx],
                       label=f'Line {idx+1}')
        else:
            ax3.axvspan(line.start_idx, len(indices) - 1, alpha=0.3, color=colors[idx],
                       label=f'Line {idx+1}')
            ax3.axvspan(0, line.end_idx, alpha=0.3, color=colors[idx])
        # 标注平均角度
        if line.start_idx <= line.end_idx:
            ax3.plot([line.start_idx, line.end_idx],
                    [line.avg_angle, line.avg_angle],
                    '--', color=colors[idx], linewidth=2)
        else:
            ax3.plot([line.start_idx, len(indices) - 1],
                    [line.avg_angle, line.avg_angle],
                    '--', color=colors[idx], linewidth=2)
            ax3.plot([0, line.end_idx],
                    [line.avg_angle, line.avg_angle],
                    '--', color=colors[idx], linewidth=2)

    ax3.set_xlabel('Contour point index', fontsize=10)
    ax3.set_ylabel('Angle (degrees)', fontsize=10)
    ax3.set_title(f'Angle Curve with {len(lines)} Line Segments', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8, ncol=min(len(lines), 4))

    # 4. 梯度曲线 + 阈值线
    ax4 = fig.add_subplot(gs[3, :])
    ax4.plot(indices, gradient_flat, 'g-', linewidth=1.5, label='Gradient')
    ax4.axhline(gradient_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Straight band = ±{gradient_threshold:.4f}')
    ax4.axhline(-gradient_threshold, color='red', linestyle='--', linewidth=2)

    # 标注直线段区间
    for idx, line in enumerate(lines):
        if line.start_idx <= line.end_idx:
            ax4.axvspan(line.start_idx, line.end_idx, alpha=0.3, color=colors[idx])
        else:
            ax4.axvspan(line.start_idx, len(indices) - 1, alpha=0.3, color=colors[idx])
            ax4.axvspan(0, line.end_idx, alpha=0.3, color=colors[idx])

    ax4.set_xlabel('Contour point index', fontsize=10)
    ax4.set_ylabel('Gradient', fontsize=10)
    ax4.set_title(
        f'Gradient Curve (|gradient| ≤ {gradient_threshold} = straight segment)',
        fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    plt.suptitle(f'Line Segment Extraction - Candidate #{candidate_idx}',
                fontsize=14, fontweight='bold')

    output_path = os.path.join(output_dir, f'candidate_{candidate_idx:02d}_lines.jpg')
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return output_path


def debug_line_extraction(image_path: str):
    """调试直线段提取"""

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return

    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = _resolve_writable_output_dir(basename)

    print(f"=== Line Segment Extraction: {basename} ===")
    print(f"Output directory: {output_dir}\n")

    # 先用灰度图提轮廓，再按轮廓内红色占比筛选
    mask = extract_red_mask_from_gray_contours(image)
    cv2.imwrite(os.path.join(output_dir, "01_red_mask.jpg"), mask)

    # 查找轮廓
    contours = find_contours(mask)
    print(f"Total contours: {len(contours)}")

    # 筛选候选轮廓
    candidates = filter_contours_by_area(contours, min_area=100, max_area=50000)
    candidates = filter_contours_by_geometry(candidates)
    candidates = filter_candidates_for_line_extraction(image, candidates)
    print(f"Candidate contours: {len(candidates)}\n")

    # 仅绘制「识别成功」的候选（角点驱动六边提取 ok），统一颜色，不区分优劣
    all_candidates_img = image.copy()
    recognized: List[Tuple[int, np.ndarray]] = []
    for idx, contour in enumerate(candidates):
        proc_preview, _src = get_proc_contour_for_line_extraction(image, contour)
        if proc_preview is None:
            continue
        corner_test = extract_line_segments_from_6_corners(proc_preview)
        if corner_test.get('ok', False):
            recognized.append((idx, contour))

    # 统一视觉：同色、同线宽；编号按识别成功顺序 1..N（括号内为候选流水号便于对照日志）
    unified_color = (0, 255, 200)
    unified_thickness = 3
    for rank, (idx, contour) in enumerate(recognized, start=1):
        cv2.drawContours(
            all_candidates_img, [contour], -1, unified_color, unified_thickness)
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(
                all_candidates_img, f"#{rank} (c{idx + 1})",
                (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(
                all_candidates_img, f"{len(contour)}pts", (cx, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    legend_y = 30
    cv2.putText(
        all_candidates_img,
        f"Recognized L: {len(recognized)} (corner-driven 6 edges)",
        (10, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, unified_color, 2)
    cv2.putText(
        all_candidates_img,
        "Filtered / failed contours are not shown",
        (10, legend_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imwrite(os.path.join(output_dir, "02_all_candidates.jpg"), all_candidates_img)

    # 逐个分析候选轮廓（通过前面筛选的候选全部处理）
    summary = []
    observations_by_candidate: Dict[int, object] = {}

    for idx, contour in enumerate(candidates):
        candidate_idx = idx + 1

        print(f"\n{'='*60}")
        print(f"Candidate #{candidate_idx}")
        print(f"Contour points: {len(contour)}")
        print(f"{'='*60}")

        proc_contour, proc_src = get_proc_contour_for_line_extraction(image, contour)
        if proc_contour is None:
            print(f"无可用分析轮廓，跳过")
            continue

        warped_image, _, warp_m = warp_to_frontal_with_matrix(image, contour)
        if warped_image is None:
            print(f"透视矫正失败，跳过")
            continue

        warped_mask = extract_red_mask_from_gray_contours(warped_image)
        warped_contours = find_contours(warped_mask)
        warped_candidates = filter_contours_by_area(
            warped_contours, min_area=50,
            max_area=warped_image.shape[0] * warped_image.shape[1])

        if proc_src == 'warped':
            proc_image = warped_image
            proc_area_ratio = cv2.contourArea(proc_contour) / float(
                warped_image.shape[0] * warped_image.shape[1])
            print(f"proc: warped 最大红轮廓, 点数: {len(proc_contour)}")
            print(f"warped 面积占比: {proc_area_ratio:.4f}")
        else:
            proc_image = image
            print(f"proc: 原图轮廓（warp 红掩膜为空恢复）, 点数: {len(proc_contour)}")
            print(f"warped 面积占比: n/a")

        cv2.imwrite(os.path.join(output_dir, f"candidate_{candidate_idx:02d}_warped.jpg"), warped_image)
        cv2.imwrite(os.path.join(output_dir, f"candidate_{candidate_idx:02d}_warped_mask.jpg"), warped_mask)

        # 角点驱动 6 边提取（主路径，替代原梯度提线流程）
        corner_driven = extract_line_segments_from_6_corners(proc_contour)
        if corner_driven.get('ok', False):
            c_chain8 = corner_driven['chain8']
            c_chain8_norm = corner_driven['chain8_normalized']
            c_lengths8 = corner_driven['lengths8']
            print(f"  [corner-driven] 6 corners: {corner_driven['corner_indices']}")
            print(f"  [corner-driven] gradient-guided fit: keep<=Q{corner_driven['grad_keep_quantile']:.2f}, min_keep_ratio={corner_driven['min_keep_ratio']:.2f}")
            print(f"  [corner-driven] chain8: {c_chain8}")
            print(f"  [corner-driven] chain8 normalized (rotation-invariant): {c_chain8_norm}")
            print(f"  [corner-driven] lengths: {c_lengths8}")

            corner_vis_path = os.path.join(
                output_dir, f"candidate_{candidate_idx:02d}_edges_from_corners.jpg")
            vis_corner_saved = visualize_corner_driven_edges(
                proc_image.copy(),
                proc_contour,
                corner_driven['corners'],
                corner_driven['lines'],
                corner_vis_path,
            )
            if vis_corner_saved:
                print(f"  [corner-driven] Visualization saved: {os.path.basename(vis_corner_saved)}")

            vis_pipeline_path = os.path.join(
                output_dir, f"candidate_{candidate_idx:02d}_corner_gradient_pipeline.jpg")
            vis_pipeline_saved = visualize_corner_gradient_pipeline(
                image=proc_image.copy(),
                contour=proc_contour,
                corners=corner_driven['corners'],
                lines=corner_driven['lines'],
                angles_flat=np.asarray(corner_driven['angles_flat']),
                gradient_flat=np.asarray(corner_driven['gradient_flat']),
                gradient_metric=np.asarray(corner_driven['gradient_metric']),
                segment_debug=corner_driven['segment_debug'],
                chain8=c_chain8,
                chain8_norm=c_chain8_norm,
                chain8_used=c_chain8,
                score=-1.0,
                output_path=vis_pipeline_path,
            )
            if vis_pipeline_saved:
                print(f"  [corner-driven] Pipeline plot saved: {os.path.basename(vis_pipeline_saved)}")

            if proc_src == 'warped' and warp_m is not None:
                inv_m = np.linalg.inv(warp_m).astype(np.float32)
                ray_dirs = _collect_backprojected_ray_dirs(corner_driven['lines'], inv_m)
                if len(ray_dirs) >= 2:
                    print(f"  [ray-hint] backprojected line rays: {len(ray_dirs)} dirs")

            ob = build_lbar_observation_from_six_corners(
                candidate_idx=candidate_idx,
                contour=contour,
                corner_driven=corner_driven,
                proc_src=proc_src,
                warp_m=warp_m,
            )
            if ob is not None:
                observations_by_candidate[candidate_idx] = ob
                msg = f"  [six-corner] K1=({ob.corner_k[0]:.1f}, {ob.corner_k[1]:.1f})"
                if ob.corner_k2 is not None:
                    msg += f"  K2=({ob.corner_k2[0]:.1f}, {ob.corner_k2[1]:.1f})"
                print(msg)

            # 验证 L 型几何约束
            is_valid, info = validate_l_shape_edges(corner_driven['lines'])
            print(f"\nL型验证: {'✓ 通过' if is_valid else '✗ 未通过'}")
            print(f"  {info}")

            summary.append({
                'candidate_idx': candidate_idx,
                'num_lines': len(corner_driven['lines']),
                'lines': corner_driven['lines'],
                'is_valid_l_shape': is_valid,
                'validation_info': info
            })
        else:
            print(f"  [corner-driven] failed: {corner_driven.get('reason', 'unknown')}")

    # 多 L 关联（原图坐标）：相邻/对顶一致性 + 最大簇 inlier + 兑换框中心启发式；tracker 进程内多帧平滑
    from calculation import l_bar_association as lba

    observations = []
    for candidate_idx in sorted(observations_by_candidate.keys()):
        observations.append(observations_by_candidate[candidate_idx])

    assoc = lba.analyze_frame_associations(observations)
    track_map = lba.get_default_l_bar_tracker().update(observations)
    assoc_vis = lba.draw_association_debug(image.copy(), assoc, track_map)
    assoc_path = os.path.join(output_dir, "03_l_bar_association.jpg")
    cv2.imwrite(assoc_path, assoc_vis)

    print(f"\n[association] six-corner observations: {len(observations)}")
    print(f"  mode: {'strict quad + center consensus' if assoc.quad_from_strict_filter else 'fallback / crowded-guard'}")
    if len(assoc.labels):
        n_clust = len(set(assoc.labels.tolist()))
        print(f"  graph components (edge threshold): {n_clust}")
        n_in = int(np.sum(assoc.inlier_mask)) if len(assoc.inlier_mask) else 0
        print(f"  inlier count (final): {n_in}")
    if assoc.quad_pts_ordered is not None:
        print(f"  quad hull: 4 edges drawn in 03_l_bar_association.jpg")
    if assoc.frame_center is not None:
        fc = assoc.frame_center
        print(f"  frame center (diag intersection heuristic): ({fc[0]:.1f}, {fc[1]:.1f})")
    print(f"  tracks matched (EMA): {len(track_map)} — see 03_l_bar_association.jpg")
    print("  (new sequence: call calculation.l_bar_association.reset_default_l_bar_tracker())")

    # 生成汇总
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total candidates analyzed: {len(candidates)}")
    print(f"Candidates with line segments: {len(summary)}")

    for item in summary:
        print(f"\nCandidate #{item['candidate_idx']}: {item['num_lines']} lines")
        for line_idx, line in enumerate(item['lines']):
            print(f"  Line {line_idx+1}: {line.length}pts @ {line.avg_angle:.0f}°")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 debug_line_extraction.py <image_path>")
        sys.exit(1)

    debug_line_extraction(sys.argv[1])
