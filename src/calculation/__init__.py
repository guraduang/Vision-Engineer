#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算模块初始化。

多 L 兑换站关联、时序稳定与绘制入口见 `l_bar_association`（已从本包 re-export）。
"""

from .chaincode import (
    get_chain_code,
    normalize_chain_code,
    simplify_chain_code,
)
from .l_shape_matcher import (
    match_l_shape_chain,
    is_l_shape_chaincode,
)
from .l_bar_association import (
    AssociationFrameResult,
    LBarObservation,
    MultiLBarTracker,
    QuadDrawStabilizer,
    analyze_frame_associations,
    classify_l_pair_k0_ray_relation,
    draw_association_debug,
    draw_yellow_quad_and_center,
    estimate_frame_center_from_four,
    get_default_l_bar_tracker,
    k0_to_opposite_corner_ray,
    observation_from_contour,
    ordered_quad_passes_k0_ring,
    reset_default_l_bar_tracker,
)
from .perspective import warp_to_frontal, warp_to_frontal_with_matrix

__all__ = [
    'get_chain_code',
    'normalize_chain_code',
    'simplify_chain_code',
    'match_l_shape_chain',
    'is_l_shape_chaincode',
    'warp_to_frontal',
    'warp_to_frontal_with_matrix',
    # 多 L 关联与可视化（算法库）
    'LBarObservation',
    'AssociationFrameResult',
    'observation_from_contour',
    'analyze_frame_associations',
    'estimate_frame_center_from_four',
    'k0_to_opposite_corner_ray',
    'classify_l_pair_k0_ray_relation',
    'ordered_quad_passes_k0_ring',
    'MultiLBarTracker',
    'get_default_l_bar_tracker',
    'reset_default_l_bar_tracker',
    'QuadDrawStabilizer',
    'draw_yellow_quad_and_center',
    'draw_association_debug',
]
