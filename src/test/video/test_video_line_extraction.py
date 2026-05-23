#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频：与 debug_line_extraction 相同的候选筛选 + 六角提线 + 多 L 关联。
输出叠加包含：预测/平滑后的黄色闭合四边形 + 框心十字，以及当前关联层选中的
L 型候选轮廓。预测器在检测短时缺失或跳变时按上一帧速度外推，面向动态场景减少断框。

默认：`data/23/2/read.avi` → `output/read_effect.mp4`。
"""

import argparse
import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from calculation.perspective import warp_to_frontal_with_matrix
from perception import (
    extract_red_mask_from_gray_contours,
    find_contours,
    filter_contours_by_area,
    filter_contours_by_geometry,
)

from calculation.l_bar_association import (
    analyze_frame_associations,
    draw_yellow_quad_and_center,
)

_DBG_PATH = os.path.join(_SRC, 'test', 'debug', 'debug_line_extraction.py')
_spec = importlib.util.spec_from_file_location('debug_line_extraction', _DBG_PATH)
dle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dle)

extract_line_segments_from_6_corners = dle.extract_line_segments_from_6_corners


def _align_quad_to_reference_local(reference: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Align cyclic/reversed quad order to the previous drawn order."""
    ref = np.asarray(reference, dtype=np.float64).reshape(4, 2)
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    candidates = []
    for arr in (q, q[::-1]):
        for shift in range(4):
            candidates.append(np.roll(arr, -shift, axis=0))
    return min(
        candidates,
        key=lambda cand: float(np.sum(np.linalg.norm(ref - cand, axis=1))),
    )


class PredictiveQuadTracker:
    """
    检测优先的四边形预测绘制器：有效 quad 到来时立即输出；短时缺失时
    使用常速度模型补偿，超过 grace 后清空，避免长时间无目标仍画框。
    """

    def __init__(
            self,
            warmup_frames: int = 1,
            min_quad_area: float = 3000.0,
            predict_grace_frames: int = 6,
            ema_alpha: float = 1.0,
            velocity_alpha: float = 0.35):
        self.warmup_frames = max(1, int(warmup_frames))
        self.min_quad_area = float(min_quad_area)
        self.predict_grace_frames = max(0, int(predict_grace_frames))
        self.ema_alpha = float(ema_alpha)
        self.velocity_alpha = float(velocity_alpha)
        self._run = 0
        self._missing = 0
        self._locked = False
        self._ema_q: Optional[np.ndarray] = None
        self._ema_c: Optional[np.ndarray] = None
        self._vel_q: Optional[np.ndarray] = None
        self._vel_c: Optional[np.ndarray] = None

    def update(
            self,
            quad: Optional[np.ndarray],
            frame_center: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        fc = None
        if frame_center is not None:
            fc = np.asarray(frame_center, dtype=np.float64).reshape(2)

        valid_q = (
            quad is not None
            and np.asarray(quad).size >= 8
            and np.asarray(quad).reshape(-1, 2).shape[0] == 4
        )
        if not valid_q:
            return self._predict()

        q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
        if abs(float(cv2.contourArea(q.astype(np.float32)))) < self.min_quad_area:
            return self._predict()
        if self._ema_q is not None:
            q = _align_quad_to_reference_local(self._ema_q, q)

        self._missing = 0
        self._run += 1

        if self._ema_q is None:
            self._ema_q = q.copy()
            self._ema_c = fc.copy() if fc is not None else None
            self._vel_q = np.zeros_like(q)
            self._vel_c = np.zeros(2, dtype=np.float64)
        else:
            old_q = self._ema_q.copy()
            old_c = self._ema_c.copy() if self._ema_c is not None else None
            a = self.ema_alpha
            self._ema_q = a * q + (1.0 - a) * self._ema_q
            dq = self._ema_q - old_q
            if self._vel_q is None:
                self._vel_q = dq
            else:
                va = self.velocity_alpha
                self._vel_q = va * dq + (1.0 - va) * self._vel_q
            if fc is not None:
                if self._ema_c is None:
                    self._ema_c = fc.copy()
                    self._vel_c = np.zeros(2, dtype=np.float64)
                else:
                    self._ema_c = a * fc + (1.0 - a) * self._ema_c
                    dc = self._ema_c - old_c if old_c is not None else np.zeros(2)
                    if self._vel_c is None:
                        self._vel_c = dc
                    else:
                        va = self.velocity_alpha
                        self._vel_c = va * dc + (1.0 - va) * self._vel_c

        if self._run < self.warmup_frames:
            return None, None, False

        self._locked = True
        return (
            self._ema_q.copy(),
            self._ema_c.copy() if self._ema_c is not None else None,
            False,
        )

    def _predict(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        if (
                not self._locked
                or self._ema_q is None
                or self._missing >= self.predict_grace_frames):
            self._missing += 1
            if self._missing > self.predict_grace_frames:
                self._locked = False
                self._run = 0
                self._ema_q = None
                self._ema_c = None
                self._vel_q = None
                self._vel_c = None
            return None, None, False
        self._missing += 1
        if self._vel_q is not None:
            self._ema_q = self._ema_q + self._vel_q
        if self._ema_c is not None and self._vel_c is not None:
            self._ema_c = self._ema_c + self._vel_c
        return (
            self._ema_q.copy(),
            self._ema_c.copy() if self._ema_c is not None else None,
            True,
        )


def draw_selected_candidate_contours(
        image: np.ndarray,
        candidates: List[np.ndarray],
        observations: List[Any],
        inlier_mask: np.ndarray) -> np.ndarray:
    """Draw contours selected by the association inlier mask."""
    vis = image.copy()
    if inlier_mask is None or len(inlier_mask) == 0:
        return vis
    for obs_idx, is_inlier in enumerate(inlier_mask):
        if not is_inlier or obs_idx >= len(observations):
            continue
        cand_idx = int(observations[obs_idx].candidate_idx)
        if cand_idx < 0 or cand_idx >= len(candidates):
            continue
        cv2.drawContours(
            vis, [candidates[cand_idx]], -1, (255, 255, 0), 2,
            lineType=cv2.LINE_AA)
        k = tuple(int(round(x)) for x in observations[obs_idx].corner_k)
        cv2.circle(vis, k, 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            vis, f'sel c{cand_idx}', (k[0] + 8, k[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return vis


def process_frame(
        frame: np.ndarray,
        quad_tracker: Optional[PredictiveQuadTracker] = None):
    """
    单帧：原图 + 稳定后的黄框/框心（或无叠加）。
    """
    vis = frame.copy()
    mask = extract_red_mask_from_gray_contours(frame)
    contours = find_contours(mask)
    candidates_geom = filter_contours_by_area(contours, min_area=100, max_area=50000)
    candidates_geom = filter_contours_by_geometry(candidates_geom)
    candidates = dle.filter_candidates_for_line_extraction(frame, candidates_geom)

    hex_packs: List[Tuple[int, Dict[str, Any], str, Optional[np.ndarray], np.ndarray]] = []
    for idx, contour in enumerate(candidates):
        proc_contour, proc_src = dle.get_proc_contour_for_line_extraction(
            frame, contour)
        if proc_contour is None:
            continue
        warped, _wc, M = warp_to_frontal_with_matrix(frame, contour)
        if warped is None or M is None:
            continue
        corner_driven = extract_line_segments_from_6_corners(proc_contour)
        if not corner_driven.get('ok', False):
            continue
        hex_packs.append((idx, corner_driven, proc_src, M, proc_contour))

    observations = []
    for cand_idx, corner_driven, proc_src, M, proc_contour in hex_packs:
        ob = dle.build_lbar_observation_from_six_corners(
            cand_idx, proc_contour, corner_driven, proc_src, M)
        if ob is not None:
            observations.append(ob)
    if observations:
        assoc = analyze_frame_associations(observations)
        qd, cc = assoc.quad_pts_ordered, assoc.frame_center
        vis = draw_selected_candidate_contours(
            vis, candidates, observations, assoc.inlier_mask)
        predicted = False
        if quad_tracker is not None:
            qd, cc, predicted = quad_tracker.update(qd, cc)
        vis = draw_yellow_quad_and_center(vis, qd, cc)
        if predicted and qd is not None:
            cv2.putText(
                vis, 'PREDICTED', (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2, cv2.LINE_AA)
    elif quad_tracker is not None:
        qd, cc, predicted = quad_tracker.update(None, None)
        vis = draw_yellow_quad_and_center(vis, qd, cc)
        if predicted and qd is not None:
            cv2.putText(
                vis, 'PREDICTED', (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2, cv2.LINE_AA)

    n_hex = len(hex_packs)
    meta = {
        'num_candidates': len(candidates),
        'num_candidates_geom': len(candidates_geom),
        'num_hex_ok': n_hex,
        'has_center': False,
    }
    return vis, meta


def process_video(
        video_path: str,
        output_path: str = None,
        max_frames: int = None,
        quad_warmup_frames: int = 1,
        quad_min_area: float = 3000.0,
        quad_predict_grace: int = 6,
        quad_ema_alpha: float = 1.0,
        quad_velocity_alpha: float = 0.35):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'错误: 无法打开视频 {video_path}')
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join('output', f'{base}_line_extraction.mp4')
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    else:
        od = os.path.dirname(os.path.abspath(output_path))
        if od:
            os.makedirs(od, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    quad_tracker: Optional[PredictiveQuadTracker] = None
    if quad_warmup_frames > 0:
        quad_tracker = PredictiveQuadTracker(
            warmup_frames=quad_warmup_frames,
            min_quad_area=quad_min_area,
            predict_grace_frames=quad_predict_grace,
            ema_alpha=quad_ema_alpha,
            velocity_alpha=quad_velocity_alpha,
        )

    frame_count = 0
    frames_with_hex = 0

    print(f'输出: {output_path}')
    if quad_tracker is not None:
        print(
            f'检测优先: 预热≥{quad_warmup_frames} 帧, 面积≥{quad_min_area}, '
            f'短丢帧补偿≤{quad_predict_grace}, '
            f'EMA={quad_ema_alpha}, 速度EMA={quad_velocity_alpha}')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_count >= max_frames:
            break

        vis, meta = process_frame(frame, quad_tracker)
        if meta.get('num_hex_ok', 0) > 0:
            frames_with_hex += 1

        out.write(vis)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f'已处理 {frame_count} 帧')

    cap.release()
    out.release()

    print(f'\n完成: 总帧数={frame_count}, 至少一个 hex_ok 的帧数={frames_with_hex}')


_DEFAULT_READ_REL = os.path.join('data', '23', '2', 'read.avi')
_DEFAULT_EFFECT_OUT_REL = os.path.join('output', 'read_effect.mp4')


def main():
    parser = argparse.ArgumentParser(
        description='视频跑通 debug_line_extraction 管线（默认 read.avi → output/read_effect.mp4）')
    parser.add_argument(
        'video_path',
        nargs='?',
        default=_DEFAULT_READ_REL,
        help=f'输入视频（默认 {_DEFAULT_READ_REL}）')
    parser.add_argument(
        '-o', '--output', default=None,
        help=f'输出 mp4（默认 {_DEFAULT_EFFECT_OUT_REL}）')
    parser.add_argument(
        '-n', '--max-frames', type=int, default=None,
        help='只处理前 n 帧（调试）')
    parser.add_argument(
        '--quad-stable-frames', type=int, default=None,
        help='兼容旧参数：作为预测器预热帧数；0=不启用预测器')
    parser.add_argument(
        '--quad-warmup-frames', type=int, default=None,
        help='预测器收到多少帧有效 quad 后开始显示（默认 1，检测到就画）')
    parser.add_argument(
        '--quad-min-area', type=float, default=3000.0,
        help='进入预测器的四边形最小面积，过滤小误框（默认 3000）')
    parser.add_argument(
        '--quad-predict-grace', type=int, default=6,
        help='已锁定后连续无 quad 时预测补偿帧数，超过清空（默认 6）')
    parser.add_argument(
        '--quad-ema-alpha', type=float, default=1.0,
        help='检测帧 quad/中心 EMA 系数；1.0 表示检测帧直接画当前框（默认 1.0）')
    parser.add_argument(
        '--quad-velocity-alpha', type=float, default=0.35,
        help='预测速度 EMA 系数（默认 0.35）')
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(_SRC, '..'))
    vp = args.video_path
    if not os.path.isabs(vp):
        cand = os.path.join(repo_root, vp)
        if os.path.isfile(cand):
            vp = cand

    out = args.output
    if not out:
        out = os.path.join(repo_root, _DEFAULT_EFFECT_OUT_REL)
    elif not os.path.isabs(out):
        out = os.path.join(repo_root, out)

    process_video(
        vp, out, args.max_frames,
        quad_warmup_frames=(
            1 if args.quad_warmup_frames is None and args.quad_stable_frames is None
            else args.quad_warmup_frames
            if args.quad_warmup_frames is not None
            else args.quad_stable_frames
        ),
        quad_min_area=args.quad_min_area,
        quad_predict_grace=args.quad_predict_grace,
        quad_ema_alpha=args.quad_ema_alpha,
        quad_velocity_alpha=args.quad_velocity_alpha,
    )


if __name__ == '__main__':
    main()
