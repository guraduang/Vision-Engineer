#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对视频时间窗内指定候选（c1、c4）输出外拐点 K 的选取诊断：
路径（外接三角形 / 链码语义覆盖 / 局部几何）、六角点、边长、链码、各凸顶点打分等。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_DBG_PATH = os.path.join(_SRC, 'test', 'debug', 'debug_line_extraction.py')
_spec = importlib.util.spec_from_file_location('debug_line_extraction', _DBG_PATH)
dle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dle)

from calculation.perspective import warp_to_frontal_with_matrix
from perception import (
    extract_red_mask_from_gray_contours,
    find_contours,
    filter_contours_by_area,
    filter_contours_by_geometry,
)


def _per_vertex_scores(
        corner_driven: Dict,
        template_variant,
        match_score: float):
    """复刻 _choose_outer_corner_from_six 中局部几何循环与语义覆盖判定（不含三角形优先）。"""
    corners = np.asarray(corner_driven.get('corners'), dtype=np.float64)
    lengths = np.asarray(corner_driven.get('lengths8'), dtype=np.float64)
    signed_area = 0.5 * float(np.sum(
        corners[:, 0] * np.roll(corners[:, 1], -1)
        - np.roll(corners[:, 0], -1) * corners[:, 1]))
    max_len = max(float(np.max(lengths)), 1.0)
    short_stub_ratio = 0.11
    stub_penalty = 0.42
    semantic_override_margin = 0.18
    semantic_idx = dle._semantic_outer_corner_from_template(template_variant)

    rows: List[Dict[str, Any]] = []
    semantic_score: Optional[float] = None
    semantic_min_adj: Optional[float] = None
    best: Optional[Tuple[float, int]] = None

    for i in range(6):
        geom_info = dle._corner_geometry_for_outer_k(corners, lengths, signed_area, i)
        row: Dict[str, Any] = {'i': i, 'role': '', 'angle': None, 'L1': None, 'L2': None,
                               'geom': None, 'angle_score': None, 'balance': None,
                               'stub': None, 'total': None}
        if geom_info is None:
            row['role'] = 'skip (concave or angle)'
            rows.append(row)
            continue
        angle, L1, L2 = geom_info
        geom = float(np.sqrt(max(L1 * L2, 0.0))) / max_len
        angle_score = 1.0 - min(abs(angle - 90.0), 90.0) / 90.0
        balance = min(L1, L2) / (max(L1, L2) + 1e-9)
        total = geom + 0.55 * angle_score + 0.18 * balance
        stub = min(L1, L2) < short_stub_ratio * max_len
        if stub:
            total -= stub_penalty
        row.update({
            'role': 'cand',
            'angle': round(angle, 2),
            'L1': round(L1, 2),
            'L2': round(L2, 2),
            'geom': round(geom, 4),
            'angle_score': round(angle_score, 4),
            'balance': round(balance, 4),
            'stub': stub,
            'total': round(total, 4),
        })
        if i == semantic_idx:
            semantic_score = total
            semantic_min_adj = min(L1, L2)
        if best is None or total > best[0]:
            best = (total, i)
        rows.append(row)

    chosen_semantic = False
    final_k: Optional[int] = None
    if (best is not None and semantic_idx is not None and semantic_score is not None
            and match_score >= 0.58 and semantic_min_adj is not None
            and semantic_min_adj >= 0.15 * max_len
            and semantic_score >= best[0] - semantic_override_margin):
        chosen_semantic = True
        final_k = semantic_idx
    elif best is not None:
        final_k = best[1]

    return rows, semantic_idx, semantic_score, semantic_min_adj, best, chosen_semantic, final_k


def _triangle_diag(corner_driven: Dict) -> Dict[str, Any]:
    corners = np.asarray(corner_driven.get('corners'), dtype=np.float64)
    lengths = np.asarray(corner_driven.get('lengths8'), dtype=np.float64)
    signed_area = 0.5 * float(np.sum(
        corners[:, 0] * np.roll(corners[:, 1], -1)
        - np.roll(corners[:, 0], -1) * corners[:, 1]))
    max_len = max(float(np.max(lengths)), 1.0)
    tri_raw = corner_driven.get('enclosing_triangle')
    out: Dict[str, Any] = {
        'tri_ok': False,
        'tri_vertices': None,
        'tri_nearest_corner': None,
        'tri_dist': None,
        'tri_reject': '',
    }
    if tri_raw is None:
        out['tri_reject'] = 'no_triangle'
        return out
    tri = np.asarray(tri_raw, dtype=np.float64)
    if tri.shape != (3, 2):
        out['tri_reject'] = 'bad_tri_shape'
        return out
    out['tri_vertices'] = np.round(tri, 2).tolist()
    best_d = 1e9
    best_ci = -1
    for tri_idx, p in enumerate(tri):
        d = np.linalg.norm(corners - p, axis=1)
        ci = int(np.argmin(d))
        dd = float(d[ci])
        if dd < best_d:
            best_d = dd
            best_ci = ci
    out['tri_nearest_corner'] = best_ci
    out['tri_dist'] = round(best_d, 3)
    thr = max(28.0, 0.24 * max_len)
    if best_d > thr:
        out['tri_reject'] = f'dist>{thr:.1f}'
        return out
    geom = dle._corner_geometry_for_outer_k(corners, lengths, signed_area, best_ci)
    if geom is None:
        out['tri_reject'] = 'corner_not_convex_elbow'
        return out
    _ang, L1, L2 = geom
    if min(L1, L2) < 0.12 * max_len:
        out['tri_reject'] = f'min_adj<{0.12*max_len:.1f}'
        return out
    out['tri_ok'] = True
    out['tri_k_idx'] = best_ci
    return out


def run_window(
        video_path: str,
        t0: float,
        t1: float,
        want_cs: List[int],
        csv_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    f0 = int(t0 * fps)
    f1 = int(t1 * fps)

    fieldnames = [
        'frame', 'time_s', 'candidate_idx', 'status', 'proc_src',
        'k_idx_final', 'k_path',
        'tri_ok', 'tri_k_idx', 'tri_dist', 'tri_nearest_v', 'tri_reject',
        'match_score', 'template', 'semantic_idx', 'geom_best_idx', 'geom_best_score',
        'semantic_override_would', 'lengths8', 'chain8', 'corners_proc_round2',
    ]
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()

        for fi in range(f0, f1 + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            mask = extract_red_mask_from_gray_contours(frame)
            cands = dle.filter_candidates_for_line_extraction(
                frame,
                filter_contours_by_geometry(
                    filter_contours_by_area(find_contours(mask), 100, 50000)))
            for cidx in want_cs:
                row_base = {'frame': fi, 'time_s': round(fi / fps, 4), 'candidate_idx': cidx}
                if cidx > len(cands) or cidx < 1:
                    w.writerow({**row_base, 'status': 'no_such_candidate', 'k_path': ''})
                    continue
                contour = cands[cidx - 1]
                proc, proc_src = dle.get_proc_contour_for_line_extraction(frame, contour)
                if proc is None:
                    w.writerow({**row_base, 'status': 'no_proc', 'k_path': ''})
                    continue
                cd = dle.extract_line_segments_from_6_corners(proc)
                if not cd.get('ok'):
                    w.writerow({
                        **row_base, 'status': 'six_corner_fail',
                        'proc_src': proc_src,
                        'k_path': '',
                    })
                    continue
                sc, tpl, *_ = dle.weighted_chaincode_match(
                    cd['chain8'], cd['lengths8'],
                    template=[0, 6, 4, 6, 4, 2], heavy_weight=2.0)
                k_final, k_src = dle._choose_outer_corner_from_six(
                    cd, template_variant=tpl, match_score=sc)
                tri_d = _triangle_diag(cd)
                _, sem_i, _, _, best_pair, sem_override, _k_geom_fallback = _per_vertex_scores(
                    cd, tpl, sc)
                k_path = k_src

                w.writerow({
                    **row_base,
                    'status': 'ok',
                    'proc_src': proc_src,
                    'k_idx_final': k_final,
                    'k_path': k_path,
                    'tri_ok': tri_d.get('tri_ok'),
                    'tri_k_idx': tri_d.get('tri_k_idx'),
                    'tri_dist': tri_d.get('tri_dist'),
                    'tri_nearest_v': tri_d.get('tri_nearest_corner'),
                    'tri_reject': tri_d.get('tri_reject', ''),
                    'match_score': round(float(sc), 4),
                    'template': str(tpl),
                    'semantic_idx': sem_i,
                    'geom_best_idx': best_pair[1] if best_pair else None,
                    'geom_best_score': round(best_pair[0], 4) if best_pair else None,
                    'semantic_override_would': sem_override,
                    'lengths8': str([int(x) for x in cd['lengths8']]),
                    'chain8': str(list(cd['chain8'])),
                    'corners_proc_round2': str(np.round(np.asarray(cd['corners']), 2).tolist()),
                })
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video_path')
    ap.add_argument('--t0', type=float, default=4.0)
    ap.add_argument('--t1', type=float, default=6.0)
    ap.add_argument('--c', type=int, nargs='+', default=[1, 4], help='candidate indices c1=1')
    ap.add_argument('-o', '--output', default='output/diag_k_pick_c1_c4_4s_6s.csv')
    args = ap.parse_args()
    repo = os.path.abspath(os.path.join(_SRC, '..'))
    vp = args.video_path
    if not os.path.isabs(vp):
        c = os.path.join(repo, vp)
        if os.path.isfile(c):
            vp = c
    run_window(vp, args.t0, args.t1, args.c, args.output)
    print('wrote', args.output)


if __name__ == '__main__':
    main()
