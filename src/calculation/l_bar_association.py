#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多 L 型灯条几何关联、误检抑制与简易时序平滑。

约束在**原图坐标系**下计算（各灯条独立 warp 后无法直接比几何）。
思路对齐 rm_vision_core「强制构造中心」：用多点/多线几何一致性投票，而非单灯条分数。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np

from calculation.corner_detector import find_l_shape_keypoints


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return (v / n).astype(np.float64)


def _line_angle_diff_deg(u: np.ndarray, v: np.ndarray) -> float:
    """无向直线夹角 [0, 90]°。"""
    c = float(np.clip(abs(np.dot(_unit(u), _unit(v))), 0.0, 1.0))
    ang = float(np.degrees(np.arccos(c)))
    return min(ang, 180.0 - ang)


def _point_to_line_dist(p: np.ndarray, origin: np.ndarray, direc: np.ndarray) -> float:
    """点 p 到过 origin、方向 direc（单位向量）的直线距离。"""
    d = _unit(direc)
    w = p.astype(np.float64) - origin.astype(np.float64)
    return float(abs(w[0] * d[1] - w[1] * d[0]))


def _intersect_lines(
        o1: np.ndarray, d1: np.ndarray,
        o2: np.ndarray, d2: np.ndarray) -> Optional[np.ndarray]:
    """两直线 o1+t d1 与 o2+s d2 的交点。"""
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float64)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t_s = np.linalg.solve(A, (o2 - o1).astype(np.float64))
    return o1 + float(t_s[0]) * d1


def _least_squares_line_center(
        origins: List[np.ndarray],
        dirs: List[np.ndarray]) -> Optional[np.ndarray]:
    """多条直线的最小二乘交点；用于相邻 L 的角平分线近似相交。"""
    if len(origins) != len(dirs) or len(origins) < 2:
        return None
    A = np.zeros((2, 2), dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    I = np.eye(2, dtype=np.float64)
    for o, d in zip(origins, dirs):
        u = _unit(d)
        P = I - np.outer(u, u)
        A += P
        b += P @ o.astype(np.float64)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    return np.linalg.solve(A, b)


@dataclass
class LBarObservation:
    """单帧、原图坐标下的一条 L 灯条观测。"""
    candidate_idx: int
    centroid: np.ndarray  # (2,) float64
    corner_k: np.ndarray  # 外拐点 K (2,)
    dir_a: np.ndarray     # K -> 端点1 单位方向
    dir_b: np.ndarray     # K -> 端点2 单位方向
    bisector: np.ndarray  # 归一化角平分线（指向 L 开口侧近似）
    area: float
    match_score: float = -1.0  # 六边链码匹配分；<0 表示未知
    arm_a_end: Optional[np.ndarray] = None
    arm_b_end: Optional[np.ndarray] = None
    # 六角提线：第一长邻边对与「余边」上第二长邻边对的第二个外角点（原图）；无则 None
    corner_k2: Optional[np.ndarray] = None

    def arm_dirs(self) -> List[np.ndarray]:
        return [self.dir_a, self.dir_b, -self.dir_a, -self.dir_b]


@dataclass
class AssociationFrameResult:
    observations: List[LBarObservation]
    pair_adjacent: np.ndarray   # n x n 对称，相邻一致性 [0,1]
    pair_opposite: np.ndarray   # n x n 对顶/对角一致性
    labels: np.ndarray          # 连通域标签
    inlier_mask: np.ndarray     # bool n — 最大优质簇成员
    frame_center: Optional[np.ndarray]  # (2,) 由 inliers 估计
    num_tracks_matched: int = 0
    # 逆时针闭合四边形顶点 (4,2)，用于绘制四条边；仅当通过四元组+中心共识筛选时有效
    quad_pts_ordered: Optional[np.ndarray] = None
    quad_from_strict_filter: bool = False


def observation_from_contour(
        candidate_idx: int,
        contour: np.ndarray) -> Optional[LBarObservation]:
    if contour is None or len(contour) < 10:
        return None
    kp = find_l_shape_keypoints(contour)
    if kp is None:
        return None
    kpts = kp["keypoints"]
    e1 = np.array(kpts[0], dtype=np.float64)
    k = np.array(kpts[1], dtype=np.float64)
    e2 = np.array(kpts[2], dtype=np.float64)
    da = _unit(e1 - k)
    db = _unit(e2 - k)
    bis = _unit(da + db)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    c = np.array([cx, cy], dtype=np.float64)
    area = float(cv2.contourArea(contour))
    return LBarObservation(
        candidate_idx=candidate_idx,
        centroid=c,
        corner_k=k,
        dir_a=da,
        dir_b=db,
        bisector=bis,
        area=area,
        match_score=-1.0,
        arm_a_end=e1,
        arm_b_end=e2,
    )


def _adjacent_consistency(
        a: LBarObservation,
        b: LBarObservation,
        parallel_deg: float = 18.0,
        max_directed_bis_deg: float = 150.0,
        max_strip_px: float = 28.0) -> float:
    """
    相邻灯条（启发式）：存在一对臂方向近似平行，且另一维上拐点落在「平行条带」内。
    """
    directed_dot = float(np.clip(np.dot(_unit(a.bisector), _unit(b.bisector)), -1.0, 1.0))
    directed_ang = float(np.degrees(np.arccos(directed_dot)))
    if directed_ang > max_directed_bis_deg:
        return 0.0

    best_par = 90.0
    best_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for u in a.arm_dirs():
        for v in b.arm_dirs():
            ang = _line_angle_diff_deg(u, v)
            if ang < best_par:
                best_par = ang
                best_pair = (u, v)
    if best_pair is None or best_par > parallel_deg:
        return 0.0
    u, _ = best_pair
    # 条带：K_b 到 K_a 处臂方向直线的距离
    d1 = _point_to_line_dist(b.corner_k, a.corner_k, u)
    d2 = _point_to_line_dist(a.corner_k, b.corner_k, u)
    strip = min(d1, d2)
    if strip > max_strip_px:
        return 0.0
    return float(np.exp(-strip / max_strip_px) * np.cos(np.radians(best_par)))


def _opposite_consistency(
        a: LBarObservation,
        b: LBarObservation,
        parallel_bis_deg: float = 22.0,
        max_directed_bis_deg: float = 150.0,
        min_sep_px: float = 35.0) -> float:
    """
    对顶灯条（启发式）：两角平分线轴线近似平行，且质心间距足够大。

    注意这里额外做有向夹角门控：若两条射线方向接近 180°，说明两枚 L 的
    中位线互相反向，不应仅因“无向平行”被拉进同一拟合组。
    """
    sep = float(np.linalg.norm(a.centroid - b.centroid))
    if sep < min_sep_px:
        return 0.0
    directed_dot = float(np.clip(np.dot(_unit(a.bisector), _unit(b.bisector)), -1.0, 1.0))
    directed_ang = float(np.degrees(np.arccos(directed_dot)))
    if directed_ang > max_directed_bis_deg:
        return 0.0
    ang = _line_angle_diff_deg(a.bisector, b.bisector)
    if ang > parallel_bis_deg:
        return 0.0
    # 质心连线与平分线方向应大致共线（对矩形框对顶 L）
    v = _unit(b.centroid - a.centroid)
    d1 = min(_line_angle_diff_deg(v, a.bisector), _line_angle_diff_deg(v, b.bisector))
    if d1 > 35.0:
        return 0.0
    return float(np.exp(-ang / parallel_bis_deg) * (1.0 - min_sep_px / (sep + min_sep_px)))


def _intersect_ray_parameters(
        o1: np.ndarray, d1: np.ndarray,
        o2: np.ndarray, d2: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float64)
    det = float(np.linalg.det(A))
    if abs(det) < 1e-9:
        return None
    t_s = np.linalg.solve(A, (o2 - o1).astype(np.float64))
    t = float(t_s[0])
    s = float(t_s[1])
    return o1 + t * d1, t, s


def k0_to_opposite_corner_ray(obs: LBarObservation) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    从 k0（corner_k）指向对径角 corner_k2 的单位方向射线 (原点, 方向)；
    无 corner_k2 或退化时返回 None。
    """
    if obs.corner_k2 is None:
        return None
    o = obs.corner_k.astype(np.float64)
    dvec = obs.corner_k2.astype(np.float64) - o
    if float(np.linalg.norm(dvec)) < 3.0:
        return None
    return o, _unit(dvec)


def _arms_one_parallel_one_antiparallel(a: LBarObservation, b: LBarObservation) -> bool:
    """
    邻边 L 型：两枚 L 的外轮廓两臂中，可匹配为「一对同向近似平行、另一对近似反向（差约 180°）」。
    允许对另一枚 L 的两臂交换次序并独立取反（无向直线 + 射线沿臂向外的符号）。
    """
    a1, a2 = _unit(a.dir_a), _unit(a.dir_b)
    b1, b2 = _unit(b.dir_a), _unit(b.dir_b)
    thr_same = 0.86
    thr_opp = -0.86
    for eb1, eb2 in ((b1, b2), (b2, b1)):
        for s1, s2 in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            u1, u2 = s1 * eb1, s2 * eb2
            if (float(np.dot(a1, u1)) > thr_same and float(np.dot(a2, u2)) < thr_opp):
                return True
            if (float(np.dot(a1, u1)) < thr_opp and float(np.dot(a2, u2)) > thr_same):
                return True
            if (float(np.dot(a1, u2)) > thr_same and float(np.dot(a2, u1)) < thr_opp):
                return True
            if (float(np.dot(a1, u2)) < thr_opp and float(np.dot(a2, u1)) > thr_same):
                return True
    return False


def _arms_both_antiparallel_pairs(a: LBarObservation, b: LBarObservation) -> bool:
    """对角 L 型：两臂与另一枚 L 的两臂均可匹配为近似反向（均约差 180°）。"""
    a1, a2 = _unit(a.dir_a), _unit(a.dir_b)
    b1, b2 = _unit(b.dir_a), _unit(b.dir_b)
    thr = -0.82
    for eb1, eb2 in ((b1, b2), (b2, b1)):
        for s1, s2 in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            u1, u2 = s1 * eb1, s2 * eb2
            if float(np.dot(a1, u1)) < thr and float(np.dot(a2, u2)) < thr:
                return True
    return False


def classify_l_pair_k0_ray_relation(
        a: LBarObservation,
        b: LBarObservation,
        *,
        ray_perp_tol_deg: float = 22.0,
        ray_opposite_dot: float = -0.88,
        max_hit_dist_px: float = 560.0,
        min_param: float = -8.0) -> Optional[Literal['adjacent', 'opposite']]:
    """
    用 k0→对径角射线 + 两臂拟合方向判别：

    - **adjacent**：两射线（无限直线）相交，且无向夹角接近 90°；同时两臂为一组同向、一组反向。
    - **opposite**：两射线方向近似相反（dot 很负）；且两臂与另一枚均为近似反向。

    若两种都勉强满足，优先 opposite（由射线 dot 更负者主导）。
    """
    ra = k0_to_opposite_corner_ray(a)
    rb = k0_to_opposite_corner_ray(b)
    if ra is None or rb is None:
        return None
    o1, d1 = ra
    o2, d2 = rb
    hit = _intersect_ray_parameters(o1, d1, o2, d2)
    if hit is None:
        return None
    p, t, s = hit
    if t < min_param or s < min_param:
        return None
    if (float(np.linalg.norm(p - o1)) > max_hit_dist_px
            or float(np.linalg.norm(p - o2)) > max_hit_dist_px):
        return None

    line_ang = _line_angle_diff_deg(d1, d2)
    perp_ok = abs(line_ang - 90.0) <= ray_perp_tol_deg
    dot_ray = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
    opp_ray_ok = dot_ray <= ray_opposite_dot

    arm_adj = _arms_one_parallel_one_antiparallel(a, b)
    arm_opp = _arms_both_antiparallel_pairs(a, b)

    cand_adj = perp_ok and arm_adj
    cand_opp = opp_ray_ok and arm_opp
    if cand_adj and cand_opp:
        return 'opposite' if dot_ray < -0.93 else 'adjacent'
    if cand_adj:
        return 'adjacent'
    if cand_opp:
        return 'opposite'
    return None


def _ccw_ordered_indices_four(
        observations: List[LBarObservation],
        idx4: Tuple[int, ...]) -> List[int]:
    """四枚 L 的外拐点绕其质心极角排序，得到 CCW 闭环顺序（与凸四边形邻接关系一致）。"""
    li = list(idx4)
    pts = np.stack([observations[i].corner_k for i in li], axis=0)
    c = np.mean(pts, axis=0)

    def ang(i: int) -> float:
        p = observations[i].corner_k.astype(np.float64) - c
        return float(np.arctan2(p[1], p[0]))

    return sorted(li, key=ang)


def _geom_k0_edges_intersect_perpendicular(
        a: LBarObservation,
        b: LBarObservation,
        *,
        ray_perp_tol_deg: float = 25.0,
        max_hit_dist_px: float = 560.0,
        min_param: float = -8.0) -> bool:
    """k0→k2 射线所在直线相交，且无向夹角接近 90°（几何项，配合两臂一致性筛误检）。"""
    ra = k0_to_opposite_corner_ray(a)
    rb = k0_to_opposite_corner_ray(b)
    if ra is None or rb is None:
        return False
    o1, d1 = ra
    o2, d2 = rb
    if _line_angle_diff_deg(d1, d2) <= 12.0:
        return False
    hit = _intersect_ray_parameters(o1, d1, o2, d2)
    if hit is None:
        return False
    p, t, s = hit
    if t < min_param or s < min_param:
        return False
    if (float(np.linalg.norm(p - o1)) > max_hit_dist_px
            or float(np.linalg.norm(p - o2)) > max_hit_dist_px):
        return False
    line_ang = _line_angle_diff_deg(d1, d2)
    return abs(line_ang - 90.0) <= ray_perp_tol_deg


def _geom_k0_rays_parallel_opposite(
        a: LBarObservation,
        b: LBarObservation,
        *,
        parallel_line_deg: float = 12.0,
        ray_opposite_dot: float = -0.85) -> bool:
    """两射线方向共线且近似反向（平行直线、无有限交点），对对径一对 L。"""
    ra = k0_to_opposite_corner_ray(a)
    rb = k0_to_opposite_corner_ray(b)
    if ra is None or rb is None:
        return False
    _o1, d1 = ra
    _o2, d2 = rb
    if _line_angle_diff_deg(d1, d2) > parallel_line_deg:
        return False
    return float(np.dot(_unit(d1), _unit(d2))) <= ray_opposite_dot


def k0_ordered_edge_ok(a: LBarObservation, b: LBarObservation) -> bool:
    """顺序环上相邻一对：射线相交且近 90°（邻边语义），或分类器已判 adjacent。"""
    if classify_l_pair_k0_ray_relation(a, b) == 'adjacent':
        return True
    return (
        _geom_k0_edges_intersect_perpendicular(a, b)
        and _arms_one_parallel_one_antiparallel(a, b))


def k0_ordered_diagonal_ok(a: LBarObservation, b: LBarObservation) -> bool:
    """对对径：射线平行且方向差约 180°（对角语义），或分类器已判 opposite。"""
    if classify_l_pair_k0_ray_relation(a, b) == 'opposite':
        return True
    return (
        _geom_k0_rays_parallel_opposite(a, b)
        and _arms_both_antiparallel_pairs(a, b))


def ordered_quad_passes_k0_ring(
        observations: List[LBarObservation],
        idx4: Tuple[int, ...]) -> bool:
    """
    将四枚 L 按 CCW 排序后：环上四条边各判「相交/近垂直」邻边关系；
    两条对角判「平行反向」对径关系。不满足则视为误入集的伪 L 或几何不一致。
    若四元中任一无 corner_k2，则无法作 k0 射线顺序校验，返回 True（兼容旧观测路径）。
    """
    if len(idx4) != 4:
        return False
    if any(observations[i].corner_k2 is None for i in idx4):
        return True
    order = _ccw_ordered_indices_four(observations, idx4)
    for t in range(4):
        ia = order[t]
        ib = order[(t + 1) % 4]
        if not k0_ordered_edge_ok(observations[ia], observations[ib]):
            return False
    if not k0_ordered_diagonal_ok(observations[order[0]], observations[order[2]]):
        return False
    if not k0_ordered_diagonal_ok(observations[order[1]], observations[order[3]]):
        return False
    return True


def _prune_inliers_to_best_ordered_quad(
        observations: List[LBarObservation],
        idxs: np.ndarray,
        comb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """在多于 4 个 inlier 时，选 comb 总分最高且通过 k0 顺序环的四元组。"""
    li = idxs.tolist()
    if len(li) < 4:
        return None
    best: Optional[Tuple[int, int, int, int]] = None
    best_sc = -1.0
    for idx4 in itertools.combinations(li, 4):
        if not ordered_quad_passes_k0_ring(observations, tuple(int(x) for x in idx4)):
            continue
        sc = _quad_pair_score(tuple(int(x) for x in idx4), comb)
        if sc > best_sc:
            best_sc = sc
            best = tuple(int(x) for x in idx4)
    return best


def _bisector_ray_consistency(
        a: LBarObservation,
        b: LBarObservation,
        min_sep_px: float = 28.0,
        max_backtrack_px: float = 12.0,
        max_ray_sep_mult: float = 3.0,
        max_ray_px: float = 360.0,
        max_directed_bis_deg: float = 150.0,
        outer_parallel_deg: float = 36.0) -> float:
    """
    两枚 L 的角平分线应在框中心附近相交，且至少一组外轮廓射线近似平行/共线。
    """
    sep = float(np.linalg.norm(a.corner_k - b.corner_k))
    if sep < min_sep_px:
        return 0.0
    directed_dot = float(np.clip(np.dot(_unit(a.bisector), _unit(b.bisector)), -1.0, 1.0))
    directed_ang = float(np.degrees(np.arccos(directed_dot)))
    if directed_ang > max_directed_bis_deg:
        return 0.0

    best_hit: Optional[Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]] = None
    for sign_a in (1.0, -1.0):
        for sign_b in (1.0, -1.0):
            d1 = _unit(a.bisector) * sign_a
            d2 = _unit(b.bisector) * sign_b
            hit = _intersect_ray_parameters(a.corner_k, d1, b.corner_k, d2)
            if hit is None:
                continue
            p, t, s = hit
            if t < -max_backtrack_px or s < -max_backtrack_px:
                continue
            ray_len = max(
                float(np.linalg.norm(p - a.corner_k)),
                float(np.linalg.norm(p - b.corner_k)),
            )
            if ray_len > max(max_ray_px, sep * max_ray_sep_mult):
                continue
            rank = ray_len + 0.25 * abs(t - s)
            if best_hit is None or rank < best_hit[1]:
                best_hit = (p, rank, abs(t - s), d1, d2)

    if best_hit is None:
        return 0.0
    _p, _rank, ray_imbalance, d1, d2 = best_hit

    best_arm = min(
        _line_angle_diff_deg(u, v)
        for u in a.arm_dirs()
        for v in b.arm_dirs()
    )
    if best_arm > outer_parallel_deg:
        return 0.0

    ray_ortho = 1.0 - abs(float(np.dot(d1, d2)))
    balance = float(np.exp(-ray_imbalance / (sep + 1.0)))
    arm_score = float(np.exp(-best_arm / outer_parallel_deg))
    return arm_score * balance * (0.45 + 0.55 * ray_ortho)


def _ray_relation_consistency(a: LBarObservation, b: LBarObservation) -> float:
    """外轮廓射线关系 + 角平分线相交关系的成对图边分数。"""
    return max(
        _adjacent_consistency(a, b),
        _bisector_ray_consistency(a, b),
    )


def _observation_quality(o: LBarObservation) -> float:
    """将六边链码匹配分映射为关系置信度；未知分数保持兼容。"""
    if o.match_score < 0.0:
        return 1.0
    return float(np.clip((o.match_score - 0.25) / 0.55, 0.0, 1.0))


def _pair_quality(a: LBarObservation, b: LBarObservation) -> float:
    return float(np.sqrt(_observation_quality(a) * _observation_quality(b)))


def _union_find(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i, j in edges:
        union(i, j)
    lab = np.zeros(n, dtype=np.int32)
    for i in range(n):
        lab[i] = find(i)
    # 压缩为 0..k-1
    uniq = {v: k for k, v in enumerate(sorted(set(lab.tolist())))}
    return np.array([uniq[v] for v in lab.tolist()], dtype=np.int32)


def estimate_frame_center_from_four(
        obs: List[LBarObservation]) -> Optional[np.ndarray]:
    """用外拐点 K 与角平分线估计兑换框中心。"""
    if len(obs) < 2:
        return None
    ks = np.stack([o.corner_k for o in obs], axis=0)
    if len(obs) == 2:
        a, b = obs[0], obs[1]
        if _opposite_consistency(a, b) > 0.0:
            return (ks[0] + ks[1]) / 2.0
        d0 = _unit(a.bisector)
        d1 = _unit(b.bisector)
        if float(np.dot(d0, b.corner_k - a.corner_k)) < 0.0:
            d0 = -d0
        if float(np.dot(d1, a.corner_k - b.corner_k)) < 0.0:
            d1 = -d1
        p = _intersect_lines(a.corner_k, d0, b.corner_k, d1)
        if p is None:
            p = _least_squares_line_center([a.corner_k, b.corner_k], [d0, d1])
        return p if p is not None else (ks[0] + ks[1]) / 2.0

    if len(obs) == 3:
        p = _least_squares_line_center(
            [o.corner_k for o in obs],
            [o.bisector for o in obs],
        )
        if p is not None and np.all(np.isfinite(p)):
            return p.astype(np.float64)

    rect = cv2.minAreaRect(ks.reshape(-1, 1, 2).astype(np.float32))
    rect_center = np.array(rect[0], dtype=np.float64)

    if len(obs) == 4 and _four_K_points_strict_convex_quad(ks):
        ordered = _order_quad_ccw_from_hull(ks)
        diag_center = _intersect_lines(
            ordered[0], ordered[2] - ordered[0],
            ordered[1], ordered[3] - ordered[1],
        )
        if diag_center is not None:
            diag_len = max(
                float(np.linalg.norm(ordered[2] - ordered[0])),
                float(np.linalg.norm(ordered[3] - ordered[1])),
            )
            if float(np.linalg.norm(diag_center - rect_center)) <= max(30.0, 0.25 * diag_len):
                return diag_center.astype(np.float64)

    return rect_center


def _four_K_points_strict_convex_quad(
        pts: np.ndarray,
        min_area: float = 120.0) -> bool:
    """四点能否构成严格凸四边形（凸包顶点数为 4）。"""
    if pts.shape != (4, 2):
        return False
    hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.float32))
    if len(hull) != 4:
        return False
    return float(cv2.contourArea(hull)) >= min_area


def _order_quad_ccw_from_hull(pts4: np.ndarray) -> np.ndarray:
    """返回逆时针闭合四顶点 (4,2)。"""
    hull = cv2.convexHull(pts4.reshape(-1, 1, 2).astype(np.float32))
    return hull.reshape(4, 2).astype(np.float64)


def _quad_internal_angles_deg(ordered: np.ndarray) -> np.ndarray:
    angles = []
    for i in range(4):
        prev_pt = ordered[(i - 1) % 4]
        cur_pt = ordered[i]
        next_pt = ordered[(i + 1) % 4]
        u = _unit(prev_pt - cur_pt)
        v = _unit(next_pt - cur_pt)
        c = float(np.clip(np.dot(u, v), -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(c))))
    return np.array(angles, dtype=np.float64)


def _quad_passes_exchange_shape(
        ordered: np.ndarray,
        angle_min_deg: float = 35.0,
        angle_max_deg: float = 145.0,
        min_side_ratio: float = 0.14,
        parallel_deg: float = 22.0,
        loose_parallel_deg: float = 38.0) -> bool:
    """近似矩形/梯形门控，剔除尖锐、极扁和无对边结构的四边形。"""
    if ordered.shape != (4, 2):
        return False
    edges = np.array([
        ordered[(i + 1) % 4] - ordered[i]
        for i in range(4)
    ], dtype=np.float64)
    lengths = np.linalg.norm(edges, axis=1)
    if float(np.min(lengths)) < 1e-6:
        return False
    if float(np.min(lengths) / np.max(lengths)) < min_side_ratio:
        return False

    angles = _quad_internal_angles_deg(ordered)
    if np.any(angles < angle_min_deg) or np.any(angles > angle_max_deg):
        return False

    par_a = _line_angle_diff_deg(edges[0], edges[2])
    par_b = _line_angle_diff_deg(edges[1], edges[3])
    if par_a <= parallel_deg and par_b <= parallel_deg:
        return True
    return min(par_a, par_b) <= parallel_deg and max(par_a, par_b) <= loose_parallel_deg


def _count_strong_pairs(
        idx4: Tuple[int, ...],
        comb: np.ndarray,
        strong_thresh: float) -> int:
    c = 0
    for ii in range(4):
        for jj in range(ii + 1, 4):
            if comb[idx4[ii], idx4[jj]] >= strong_thresh:
                c += 1
    return c


def _count_strong_pairs_matrix(
        idx4: Tuple[int, ...],
        mat: np.ndarray,
        strong_thresh: float) -> int:
    c = 0
    for ii in range(4):
        for jj in range(ii + 1, 4):
            if mat[idx4[ii], idx4[jj]] >= strong_thresh:
                c += 1
    return c


def _quad_pair_score(idx4: Tuple[int, ...], comb: np.ndarray) -> float:
    return float(sum(
        comb[idx4[ii], idx4[jj]]
        for ii in range(4) for jj in range(ii + 1, 4)))


def _enumerate_valid_exchange_quads(
        observations: List[LBarObservation],
        comb: np.ndarray,
        adj: np.ndarray,
        opp: np.ndarray,
        min_strong_pairs: int = 2,
        min_adjacent_strong_pairs: int = 2,
        min_opposite_strong_pairs: int = 1,
        strong_pair_thresh: float = 0.30,
        min_quad_area: float = 120.0,
        min_quad_match_score: float = 0.30,
        quad_angle_min_deg: float = 35.0,
        quad_angle_max_deg: float = 145.0,
        min_quad_side_ratio: float = 0.14,
        quad_parallel_deg: float = 22.0,
        quad_loose_parallel_deg: float = 38.0,
        center_inside_quad: bool = True) -> List[Tuple[float, Tuple[int, int, int, int], np.ndarray, np.ndarray]]:
    """
    枚举：外拐点构成凸四边形 + 至少 min_strong_pairs 对强 comb；
    且强边中须同时含足够「相邻」（臂/条带）与「对顶」（平分线/质心）两类，避免仅靠偶然 comb 的假四元组。
    返回列表 (score, idx4, center, quad_ordered)。
    """
    n = len(observations)
    out: List[Tuple[float, Tuple[int, int, int, int], np.ndarray, np.ndarray]] = []
    if n < 4:
        return out

    for idx4 in itertools.combinations(range(n), 4):
        if any(
                0.0 <= observations[i].match_score < min_quad_match_score
                for i in idx4):
            continue
        pts = np.stack([observations[i].corner_k for i in idx4], axis=0)
        if not _four_K_points_strict_convex_quad(pts, min_area=min_quad_area):
            continue
        ordered = _order_quad_ccw_from_hull(pts)
        if not _quad_passes_exchange_shape(
                ordered,
                angle_min_deg=quad_angle_min_deg,
                angle_max_deg=quad_angle_max_deg,
                min_side_ratio=min_quad_side_ratio,
                parallel_deg=quad_parallel_deg,
                loose_parallel_deg=quad_loose_parallel_deg):
            continue
        n_strong = _count_strong_pairs(idx4, comb, strong_pair_thresh)
        if n_strong < min_strong_pairs:
            continue
        n_adj_s = _count_strong_pairs_matrix(idx4, adj, strong_pair_thresh)
        n_opp_s = _count_strong_pairs_matrix(idx4, opp, strong_pair_thresh)
        if n_adj_s < min_adjacent_strong_pairs or n_opp_s < min_opposite_strong_pairs:
            continue
        sub = [observations[i] for i in idx4]
        center = estimate_frame_center_from_four(sub)
        if center is None:
            continue
        if center_inside_quad:
            pi = (float(center[0]), float(center[1]))
            dist_in = cv2.pointPolygonTest(
                ordered.astype(np.float32).reshape(-1, 1, 2),
                pi, True,
            )
            if dist_in < -4.0:
                continue
        if not ordered_quad_passes_k0_ring(observations, tuple(int(i) for i in idx4)):
            continue
        score = _quad_pair_score(idx4, comb) + 0.4 * float(n_strong)
        out.append((score, tuple(idx4), center.astype(np.float64), ordered))
    return out


def _cluster_mean_center(cl: Dict) -> np.ndarray:
    arr = np.stack([t[2] for t in cl["items"]], axis=0)
    return np.mean(arr, axis=0)


def _consensus_pick_best_quad(
        quads: List[Tuple[float, Tuple[int, int, int, int], np.ndarray, np.ndarray]],
        center_cluster_px: float = 48.0,
        ambiguous_second_cluster_frac: float = 0.52,
        ambiguous_center_sep_mult: float = 1.75) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], float]]:
    """
    多组合法四边形：按估计框心聚类，只保留「中心扎堆」的主簇；簇内取 score 最高。
    若第二大簇得分接近主簇且两簇平均框心明显分离，视为歧义，返回 None（不强行选一边）。
    """
    if not quads:
        return None
    # 按框心聚类（简单并查集 / 贪心簇）
    quads_sorted = sorted(quads, key=lambda x: -x[0])
    clusters: List[Dict] = []
    for sc, idx4, cc, qord in quads_sorted:
        placed = False
        for cl in clusters:
            rep = cl["centers"]
            # 与簇内任一心足够近则并入
            if any(float(np.linalg.norm(cc - c)) < center_cluster_px for c in rep):
                cl["items"].append((sc, idx4, cc, qord))
                cl["centers"].append(cc)
                placed = True
                break
        if not placed:
            clusters.append({"items": [(sc, idx4, cc, qord)], "centers": [cc]})

    def cluster_strength(cl: Dict) -> float:
        return sum(t[0] for t in cl["items"])

    clusters_ranked = sorted(clusters, key=cluster_strength, reverse=True)
    if len(clusters_ranked) >= 2:
        s0 = cluster_strength(clusters_ranked[0])
        s1 = cluster_strength(clusters_ranked[1])
        if s0 > 1e-6 and s1 >= ambiguous_second_cluster_frac * s0:
            dcc = float(np.linalg.norm(
                _cluster_mean_center(clusters_ranked[0])
                - _cluster_mean_center(clusters_ranked[1])))
            if dcc > center_cluster_px * ambiguous_center_sep_mult:
                return None

    best_cluster = clusters_ranked[0]
    # 主簇内最优四元组
    best_item = max(best_cluster["items"], key=lambda t: t[0])
    sc, idx4, cc, qord = best_item
    return (cc, qord, idx4, sc)


def _fallback_union_find_inliers(
        n: int,
        comb: np.ndarray,
        edge_thresh: float) -> Tuple[np.ndarray, int]:
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if comb[i, j] >= edge_thresh:
                edges.append((i, j))
    labels = _union_find(n, edges)
    best_lab = -1
    best_score = -1.0
    for lab in sorted(set(labels.tolist())):
        idxs = np.where(labels == lab)[0]
        if len(idxs) < 2:
            continue
        s = sum(
            comb[idxs[ii], idxs[jj]]
            for ii in range(len(idxs)) for jj in range(ii + 1, len(idxs)))
        bonus = 0.15 * max(0.0, min(len(idxs), 4) - 2)
        if s + bonus > best_score:
            best_score = s + bonus
            best_lab = int(lab)
    inlier = np.zeros(n, dtype=bool)
    if best_lab >= 0:
        inlier = labels == best_lab
    return inlier, best_lab


def _quad_from_inlier_K_points(
        observations: List[LBarObservation],
        inlier: np.ndarray,
        min_area: float = 80.0) -> Optional[np.ndarray]:
    """回退路径：恰好 4 个 inlier 且四枚 K 构成凸四边形时，返回 CCW 四顶点。"""
    idxs = np.where(inlier)[0]
    if len(idxs) != 4:
        return None
    pts = np.stack([observations[i].corner_k for i in idxs], axis=0)
    if not _four_K_points_strict_convex_quad(pts, min_area=min_area):
        return None
    return _order_quad_ccw_from_hull(pts)


def _center_inside_expanded_k_bounds(
        center: Optional[np.ndarray],
        observations: List[LBarObservation],
        expand_min_px: float = 80.0,
        expand_diag_frac: float = 0.65) -> bool:
    if center is None or len(observations) < 2:
        return False
    pts = np.stack([o.corner_k for o in observations], axis=0)
    lo = np.min(pts, axis=0)
    hi = np.max(pts, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    expand = max(expand_min_px, expand_diag_frac * diag)
    return bool(
        lo[0] - expand <= center[0] <= hi[0] + expand
        and lo[1] - expand <= center[1] <= hi[1] + expand
    )


def analyze_frame_associations(
        observations: List[LBarObservation],
        edge_thresh: float = 0.28,
        strict_exchange_quad: bool = True,
        min_strong_pairs: int = 2,
        min_adjacent_strong_pairs: int = 2,
        min_opposite_strong_pairs: int = 1,
        strong_pair_thresh: float = 0.30,
        center_cluster_px: float = 48.0,
        min_quad_area: float = 120.0,
        min_quad_match_score: float = 0.30,
        quad_angle_min_deg: float = 35.0,
        quad_angle_max_deg: float = 145.0,
        min_quad_side_ratio: float = 0.14,
        quad_parallel_deg: float = 22.0,
        quad_loose_parallel_deg: float = 38.0,
        ambiguous_second_cluster_frac: float = 0.52,
        ambiguous_center_sep_mult: float = 1.75) -> AssociationFrameResult:
    """
    strict_exchange_quad=True（默认）：
      只接受「凸四边形 + 强 comb + 强射线关系/强对顶均达标 + 帧中心在形内」的四元组；
      多组时按框心聚类；主簇与次强簇得分接近且平均框心分离则整帧放弃严格结果。
    若无合法四元组，用射线关系图的最可信连通分量估计中心。
    """
    n = len(observations)
    if n == 0:
        return AssociationFrameResult(
            observations=[],
            pair_adjacent=np.zeros((0, 0)),
            pair_opposite=np.zeros((0, 0)),
            labels=np.zeros(0, dtype=np.int32),
            inlier_mask=np.zeros(0, dtype=bool),
            frame_center=None,
            quad_pts_ordered=None,
            quad_from_strict_filter=False,
        )

    adj = np.zeros((n, n), dtype=np.float64)
    opp = np.zeros((n, n), dtype=np.float64)
    _k0_ray_floor = 0.58
    for i in range(n):
        for j in range(i + 1, n):
            a, b = observations[i], observations[j]
            q = _pair_quality(a, b)
            adj[i, j] = adj[j, i] = _ray_relation_consistency(a, b) * q
            opp[i, j] = opp[j, i] = _opposite_consistency(a, b) * q
            rel = classify_l_pair_k0_ray_relation(a, b)
            if rel == 'adjacent':
                adj[i, j] = adj[j, i] = max(adj[i, j], _k0_ray_floor)
            elif rel == 'opposite':
                opp[i, j] = opp[j, i] = max(opp[i, j], _k0_ray_floor)
    comb = np.maximum(adj, opp)

    inlier = np.zeros(n, dtype=bool)
    center: Optional[np.ndarray] = None
    quad_ordered: Optional[np.ndarray] = None
    strict_used = False
    labels = np.zeros(n, dtype=np.int32)

    if strict_exchange_quad and n >= 4:
        adj_need = min_adjacent_strong_pairs
        opp_need = min_opposite_strong_pairs
        if n > 4:
            adj_need = min(adj_need, 1)
            opp_need = max(opp_need, 2)
        valid = _enumerate_valid_exchange_quads(
            observations,
            comb,
            adj,
            opp,
            min_strong_pairs=min_strong_pairs,
            min_adjacent_strong_pairs=adj_need,
            min_opposite_strong_pairs=opp_need,
            strong_pair_thresh=strong_pair_thresh,
            min_quad_area=min_quad_area,
            min_quad_match_score=min_quad_match_score,
            quad_angle_min_deg=quad_angle_min_deg,
            quad_angle_max_deg=quad_angle_max_deg,
            min_quad_side_ratio=min_quad_side_ratio,
            quad_parallel_deg=quad_parallel_deg,
            quad_loose_parallel_deg=quad_loose_parallel_deg,
            center_inside_quad=True,
        )
        picked = _consensus_pick_best_quad(
            valid,
            center_cluster_px=center_cluster_px,
            ambiguous_second_cluster_frac=ambiguous_second_cluster_frac,
            ambiguous_center_sep_mult=ambiguous_center_sep_mult,
        )
        if picked is not None:
            center, quad_ordered, idx4, _sc = picked
            strict_used = True
            for i in idx4:
                inlier[i] = True
            # 多候选时：若框内/边上的额外轮廓与所选角强关联，说明候选竞争仍未消解。
            if n > 4:
                idx4_set = set(idx4)
                for j in range(n):
                    if j in idx4_set:
                        continue
                    dist_in = cv2.pointPolygonTest(
                        quad_ordered.astype(np.float32).reshape(-1, 1, 2),
                        (float(observations[j].corner_k[0]), float(observations[j].corner_k[1])),
                        True,
                    )
                    if dist_in >= -4.0 and max(comb[j, i] for i in idx4_set) >= strong_pair_thresh:
                        strict_used = False
                        inlier = np.zeros(n, dtype=bool)
                        center = None
                        quad_ordered = None
                        break

    if not strict_used:
        # 严格四边形失败时，使用射线关系图的最可信连通分量估计中心。
        inlier, _best_lab = _fallback_union_find_inliers(n, comb, edge_thresh)
        idxs = np.where(inlier)[0]
        if len(idxs) > 4:
            pr = _prune_inliers_to_best_ordered_quad(observations, idxs, comb)
            inlier = np.zeros(n, dtype=bool)
            if pr is not None:
                for i in pr:
                    inlier[i] = True
            idxs = np.where(inlier)[0]
        elif len(idxs) == 4:
            if not ordered_quad_passes_k0_ring(
                    observations, tuple(int(i) for i in idxs.tolist())):
                inlier = np.zeros(n, dtype=bool)
                idxs = np.where(inlier)[0]
        sub = [observations[i] for i in range(n) if inlier[i]]
        quad_ordered = _quad_from_inlier_K_points(observations, inlier)
        if quad_ordered is not None:
            center = estimate_frame_center_from_four(sub)
        elif len(sub) >= 2:
            center = estimate_frame_center_from_four(sub)
            if not _center_inside_expanded_k_bounds(center, sub):
                center = None
                inlier = np.zeros(n, dtype=bool)
        else:
            center = None
            inlier = np.zeros(n, dtype=bool)

    # 用于调试：全图 pairwise 标签（严格模式下仍标出图结构）
    edges_fb: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if comb[i, j] >= edge_thresh:
                edges_fb.append((i, j))
    labels = _union_find(n, edges_fb)

    return AssociationFrameResult(
        observations=observations,
        pair_adjacent=adj,
        pair_opposite=opp,
        labels=labels,
        inlier_mask=inlier,
        frame_center=center,
        quad_pts_ordered=quad_ordered,
        quad_from_strict_filter=strict_used,
    )


@dataclass
class _TrackState:
    track_id: int
    centroid_ema: np.ndarray
    hits: int = 1
    # 外拐点 K 与链码分 EMA，用于时序稳定性门控（拟合前筛掉跳变/暴跌帧）
    k_ema: Optional[np.ndarray] = None
    match_score_ema: float = -1.0


class MultiLBarTracker:
    """
    简易时序跟踪：按质心最近邻匹配，EMA 平滑；用于减轻单帧误检闪烁。
    多帧调用同一进程内复用同一 tracker 实例即可。

    `observation_stability_mask`：在 `update` 之后调用，仅当轨迹命中次数足够、
    当前 K 相对 k_ema 位移不过大、链码分相对 match_score_ema 未暴跌时判为稳定，
    稳定帧才参与多 L 拟合；异常帧不更新 k_ema，避免单帧错点污染。
    """

    def __init__(
            self,
            max_match_px: float = 95.0,
            ema_alpha: float = 0.35,
            miss_prune: int = 4,
            stability_min_hits: int = 3,
            max_k_jump_px: float = 48.0,
            score_drop_thresh: float = 0.22):
        self.max_match_px = max_match_px
        self.ema_alpha = ema_alpha
        self.miss_prune = miss_prune
        self.stability_min_hits = stability_min_hits
        self.max_k_jump_px = max_k_jump_px
        self.score_drop_thresh = score_drop_thresh
        self._tracks: Dict[int, _TrackState] = {}
        self._miss: Dict[int, int] = {}
        self._next_id = 1

    def update(
            self,
            observations: List[LBarObservation]) -> Dict[int, int]:
        """
        Returns:
            candidate_idx -> track_id
        """
        assigned: Dict[int, int] = {}
        if not observations:
            for tid in list(self._tracks.keys()):
                self._miss[tid] = self._miss.get(tid, 0) + 1
                if self._miss[tid] >= self.miss_prune:
                    del self._tracks[tid]
                    del self._miss[tid]
            return assigned

        used_tracks = set()
        for obs in observations:
            best_tid = None
            best_d = self.max_match_px + 1.0
            for tid, st in self._tracks.items():
                if tid in used_tracks:
                    continue
                d = float(np.linalg.norm(obs.centroid - st.centroid_ema))
                if d < best_d:
                    best_d = d
                    best_tid = tid
            if best_tid is not None and best_d <= self.max_match_px:
                st = self._tracks[best_tid]
                a = self.ema_alpha
                st.centroid_ema = (1.0 - a) * st.centroid_ema + a * obs.centroid
                st.hits += 1
                self._miss[best_tid] = 0
                used_tracks.add(best_tid)
                assigned[obs.candidate_idx] = best_tid
            else:
                tid = self._next_id
                self._next_id += 1
                ms0 = float(obs.match_score) if obs.match_score >= 0.0 else 0.55
                self._tracks[tid] = _TrackState(
                    track_id=tid,
                    centroid_ema=obs.centroid.copy(),
                    k_ema=obs.corner_k.astype(np.float64).copy(),
                    match_score_ema=ms0,
                )
                self._miss[tid] = 0
                used_tracks.add(tid)
                assigned[obs.candidate_idx] = tid

        for tid in list(self._tracks.keys()):
            if tid not in used_tracks:
                self._miss[tid] = self._miss.get(tid, 0) + 1
                if self._miss[tid] >= self.miss_prune:
                    del self._tracks[tid]
                    del self._miss[tid]
        return assigned

    def observation_stability_mask(
            self,
            observations: List[LBarObservation],
            track_map: Dict[int, int]) -> List[bool]:
        """
        须在每帧 `update(observations)` 之后调用。
        返回与 observations 等长的布尔列表：True 表示该观测可参与 `analyze_frame_associations`。
        """
        a = self.ema_alpha
        out: List[bool] = []
        for obs in observations:
            tid = track_map.get(obs.candidate_idx)
            if tid is None or tid not in self._tracks:
                out.append(False)
                continue
            st = self._tracks[tid]
            if st.k_ema is None:
                st.k_ema = obs.corner_k.astype(np.float64).copy()
            if st.match_score_ema < 0.0 and obs.match_score >= 0.0:
                st.match_score_ema = float(obs.match_score)

            k = obs.corner_k.astype(np.float64)
            if st.hits < self.stability_min_hits:
                st.k_ema = (1.0 - a) * st.k_ema + a * k
                if obs.match_score >= 0.0:
                    st.match_score_ema = (
                        (1.0 - a) * st.match_score_ema + a * float(obs.match_score))
                out.append(False)
                continue

            dk = float(np.linalg.norm(k - st.k_ema))
            stable = dk <= self.max_k_jump_px
            if obs.match_score >= 0.0 and st.match_score_ema >= 0.0:
                if obs.match_score < st.match_score_ema - self.score_drop_thresh:
                    stable = False
            if stable:
                st.k_ema = (1.0 - a) * st.k_ema + a * k
                if obs.match_score >= 0.0:
                    st.match_score_ema = (
                        (1.0 - a) * st.match_score_ema + a * float(obs.match_score))
            out.append(stable)
        return out


_default_tracker: Optional[MultiLBarTracker] = None


def get_default_l_bar_tracker() -> MultiLBarTracker:
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = MultiLBarTracker()
    return _default_tracker


def reset_default_l_bar_tracker() -> None:
    global _default_tracker
    _default_tracker = MultiLBarTracker()


def _quad_max_corner_dist_cyclic(a: np.ndarray, b: np.ndarray) -> float:
    """两枚 CCW 四顶点 (4,2)，允许循环移位与顺逆序对齐，取最小 max 顶点距。"""
    a = np.asarray(a, dtype=np.float64).reshape(4, 2)
    b = np.asarray(b, dtype=np.float64).reshape(4, 2)
    best = float('inf')
    for b_alt in (b, np.flipud(b).copy()):
        for s in range(4):
            br = np.roll(b_alt, -s, axis=0)
            best = min(best, float(np.max(np.linalg.norm(a - br, axis=1))))
    return best


def _align_quad_to_reference(ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    """将 q 循环/翻转对齐到 ref，使 max 顶点距最小；返回对齐后的 (4,2)。"""
    ref = np.asarray(ref, dtype=np.float64).reshape(4, 2)
    q = np.asarray(q, dtype=np.float64).reshape(4, 2)
    best = q.copy()
    best_d = float('inf')
    for b_alt in (q, np.flipud(q).copy()):
        for s in range(4):
            cand = np.roll(b_alt, -s, axis=0)
            d = float(np.max(np.linalg.norm(ref - cand, axis=1)))
            if d < best_d:
                best_d = d
                best = cand.copy()
    return best


class QuadDrawStabilizer:
    """
    兑换框四边形 + 框心时序门控：连续多帧几何一致后才「上屏」；
    锁定后对顶点做 EMA，并在短时丢检测、单帧跳变时保持上一帧显示，减轻闪烁。
    """

    def __init__(
            self,
            min_consecutive_frames: int = 12,
            max_corner_jump_px: float = 32.0,
            miss_grace_frames: int = 6,
            jump_break_grace_frames: int = 4,
            ema_alpha: float = 0.38):
        self.min_consecutive_frames = max(1, int(min_consecutive_frames))
        self.max_corner_jump_px = float(max_corner_jump_px)
        self.miss_grace_frames = max(0, int(miss_grace_frames))
        self.jump_break_grace_frames = max(1, int(jump_break_grace_frames))
        self.ema_alpha = float(ema_alpha)
        self._prev: Optional[np.ndarray] = None
        self._stable_run = 0
        self._locked = False
        self._ema_q: Optional[np.ndarray] = None
        self._ema_c: Optional[np.ndarray] = None
        self._miss = 0
        self._jump_bad = 0

    def reset(self) -> None:
        self._prev = None
        self._stable_run = 0
        self._locked = False
        self._ema_q = None
        self._ema_c = None
        self._miss = 0
        self._jump_bad = 0

    def update(
            self,
            quad: Optional[np.ndarray],
            frame_center: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        返回 (quad_draw, center_draw)；未稳定或丢失且无 grace 时为 (None, None)。
        """
        fc = None
        if frame_center is not None:
            fc = np.asarray(frame_center, dtype=np.float64).reshape(2)

        bad_q = (
            quad is None or np.asarray(quad).size < 8
            or np.asarray(quad).reshape(-1, 2).shape[0] != 4)

        if bad_q:
            self._prev = None
            self._stable_run = 0
            self._jump_bad = 0
            self._miss += 1
            if (
                    self._locked
                    and self._ema_q is not None
                    and self._miss <= self.miss_grace_frames):
                return (
                    self._ema_q.copy(),
                    self._ema_c.copy() if self._ema_c is not None else None,
                )
            if self._miss > self.miss_grace_frames:
                self._locked = False
                self._ema_q = None
                self._ema_c = None
            return None, None

        self._miss = 0
        q = np.asarray(quad, dtype=np.float64).reshape(4, 2)

        if self._prev is None:
            self._prev = q.copy()
            self._stable_run = 1
            self._jump_bad = 0
            return self._try_promote_lock(fc, q)

        qa = _align_quad_to_reference(self._prev, q)
        d = float(np.max(np.linalg.norm(self._prev - qa, axis=1)))

        if d > self.max_corner_jump_px:
            self._jump_bad += 1
            if (
                    self._locked
                    and self._ema_q is not None
                    and self._jump_bad <= self.jump_break_grace_frames):
                return (
                    self._ema_q.copy(),
                    self._ema_c.copy() if self._ema_c is not None else None,
                )
            self._locked = False
            self._ema_q = None
            self._ema_c = None
            self._prev = qa.copy()
            self._stable_run = 1
            self._jump_bad = 0
            return None, None

        self._jump_bad = 0
        self._stable_run += 1
        self._prev = qa.copy()

        if not self._locked:
            return self._try_promote_lock(fc, qa)

        a = self.ema_alpha
        if self._ema_q is None:
            self._ema_q = qa.copy()
        else:
            self._ema_q = a * qa + (1.0 - a) * self._ema_q
        if fc is not None:
            if self._ema_c is None:
                self._ema_c = fc.copy()
            else:
                self._ema_c = a * fc + (1.0 - a) * self._ema_c
        return self._ema_q.copy(), (
            self._ema_c.copy() if self._ema_c is not None else None)

    def _try_promote_lock(
            self,
            fc: Optional[np.ndarray],
            qa: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._stable_run < self.min_consecutive_frames:
            return None, None
        self._locked = True
        self._ema_q = qa.copy()
        self._ema_c = fc.copy() if fc is not None else None
        return self._ema_q.copy(), (
            self._ema_c.copy() if self._ema_c is not None else None)


def draw_yellow_quad_and_center(
        image: np.ndarray,
        quad_pts_ordered: Optional[np.ndarray],
        frame_center: Optional[np.ndarray],
        *,
        yellow_bgr: Tuple[int, int, int] = (0, 255, 255),
        quad_thickness: int = 3,
        cross_size: int = 16) -> np.ndarray:
    """仅绘制黄色闭合四边形与框心十字（BGR 黄 = 高 G/R）。"""
    vis = image.copy()
    if quad_pts_ordered is not None and np.asarray(quad_pts_ordered).size >= 8:
        q = np.round(np.asarray(quad_pts_ordered)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            vis, [q], isClosed=True, color=yellow_bgr, thickness=quad_thickness,
            lineType=cv2.LINE_AA)
    if frame_center is not None:
        fc = np.asarray(frame_center, dtype=np.float64).reshape(2)
        pt = tuple(int(round(x)) for x in fc)
        cv2.drawMarker(
            vis, pt, yellow_bgr, cv2.MARKER_CROSS, cross_size, 2, cv2.LINE_AA)
    return vis


def draw_association_debug(
        image: np.ndarray,
        result: AssociationFrameResult,
        track_map: Optional[Dict[int, int]] = None) -> np.ndarray:
    vis = image.copy()
    obs = result.observations
    n = len(obs)

    # 兑换框四条边：外拐点凸包闭合多边形（与透视倾斜无关，始终在图像平面）
    if result.quad_pts_ordered is not None:
        q = np.round(result.quad_pts_ordered).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            vis, [q], isClosed=True, color=(0, 220, 255), thickness=3,
            lineType=cv2.LINE_AA)

    for i, o in enumerate(obs):
        color = (0, 255, 120) if (i < len(result.inlier_mask) and result.inlier_mask[i]) else (100, 100, 255)
        c = tuple(int(round(x)) for x in o.centroid)
        cv2.circle(vis, c, 6, color, 2)
        k = tuple(int(round(x)) for x in o.corner_k)
        is_in = i < len(result.inlier_mask) and result.inlier_mask[i]
        cv2.circle(vis, k, 5 if is_in else 4, color, -1)
        if is_in:
            cv2.circle(vis, k, 10, (0, 255, 255), 2, cv2.LINE_AA)
        arm_th = 4 if is_in else 2
        arm_col = (0, 220, 255) if is_in else (255, 180, 0)
        if o.arm_a_end is not None:
            cv2.line(
                vis, k,
                tuple(int(round(x)) for x in o.arm_a_end.reshape(2)),
                arm_col, arm_th, cv2.LINE_AA,
            )
        if o.arm_b_end is not None:
            cv2.line(
                vis, k,
                tuple(int(round(x)) for x in o.arm_b_end.reshape(2)),
                arm_col, arm_th, cv2.LINE_AA,
            )
        rk = k0_to_opposite_corner_ray(o)
        if rk is not None:
            ro, rd = rk
            ray_len = 92.0
            p_end = ro + rd * ray_len
            cv2.line(
                vis,
                tuple(int(round(x)) for x in ro),
                tuple(int(round(x)) for x in p_end),
                (60, 200, 120), 2, cv2.LINE_AA,
            )
        if o.corner_k2 is not None:
            k2 = tuple(int(round(x)) for x in o.corner_k2.reshape(2))
            cv2.circle(vis, k2, 5, (255, 0, 220), -1, cv2.LINE_AA)
            cv2.circle(vis, k2, 8, (200, 200, 255), 1, cv2.LINE_AA)
        lab = f"c{o.candidate_idx}"
        if track_map and o.candidate_idx in track_map:
            lab += f"/t{track_map[o.candidate_idx]}"
        cv2.putText(vis, lab, (c[0] + 6, c[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 中心方向：仅作为辅助线，外轮廓拟合线已在每个 K 的两条邻边上绘制。
    if result.frame_center is not None:
        fc = result.frame_center.reshape(2).astype(np.float64)
        for i, o in enumerate(obs):
            if i >= len(result.inlier_mask) or not result.inlier_mask[i]:
                continue
            k = o.corner_k.astype(np.float64)
            v = fc - k
            ln = float(np.linalg.norm(v))
            if ln < 1e-6:
                continue
            step = min(ln, 60.0)
            p2 = k + (v / ln) * step
            cv2.line(
                vis,
                tuple(int(round(x)) for x in k),
                tuple(int(round(x)) for x in p2),
                (120, 120, 120), 1, cv2.LINE_AA,
            )

    # k0 射线语义成对：邻边 L（黄青）/ 对角 L（紫）
    for i in range(n):
        for j in range(i + 1, n):
            rel = classify_l_pair_k0_ray_relation(obs[i], obs[j])
            if rel is None:
                continue
            kia = tuple(int(round(x)) for x in obs[i].corner_k)
            kjb = tuple(int(round(x)) for x in obs[j].corner_k)
            col = (0, 220, 255) if rel == 'adjacent' else (255, 0, 200)
            cv2.line(vis, kia, kjb, col, 1, cv2.LINE_AA)

    # 仅 inlier 之间的弱连线（辅助看关系；主结构以四边形为准）
    for i in range(n):
        for j in range(i + 1, n):
            if i >= len(result.inlier_mask) or j >= len(result.inlier_mask):
                continue
            if not (result.inlier_mask[i] and result.inlier_mask[j]):
                continue
            w = max(result.pair_adjacent[i, j], result.pair_opposite[i, j])
            if w < 0.12:
                continue
            p1 = tuple(int(round(x)) for x in obs[i].centroid)
            p2 = tuple(int(round(x)) for x in obs[j].centroid)
            col = (90, int(120 * w), int(120 * w))
            cv2.line(vis, p1, p2, col, 1, cv2.LINE_AA)

    if result.frame_center is not None:
        fc = tuple(int(round(x)) for x in result.frame_center.reshape(2))
        cv2.drawMarker(vis, fc, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(vis, "frame center", (fc[0] + 8, fc[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 四 inlier 时标注 k0 顺序环（CCW 0..3），与 ordered_quad_passes_k0_ring 一致
    if n >= 4 and int(np.sum(result.inlier_mask)) == 4:
        idxs4 = tuple(i for i in range(n) if result.inlier_mask[i])
        if len(idxs4) == 4:
            order = _ccw_ordered_indices_four(obs, idxs4)
            for rank, ii in enumerate(order):
                k = tuple(int(round(x)) for x in obs[ii].corner_k)
                cv2.putText(
                    vis, str(rank), (k[0] + 5, k[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 80), 2, cv2.LINE_AA)

    y = 24
    mode = "strict quad+consensus" if result.quad_from_strict_filter else "fallback cluster"
    cv2.putText(vis, mode, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)
    return vis
