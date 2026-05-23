#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本 - 视频 L 型检测
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


def process_video(video_path: str, output_path: str = None):
    """
    处理视频，检测 L 型灯条并输出可视化视频

    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height} @ {fps}fps")
    if total_frames > 0:
        print(f"总帧数: {total_frames}")
    else:
        print("总帧数: 未知（某些视频格式无法获取）")

    # 设置输出路径
    if output_path is None:
        os.makedirs('output', exist_ok=True)
        basename = os.path.basename(video_path).split('.')[0]
        output_path = f"output/{basename}_detected.mp4"

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0

    print("开始处理视频...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # 提取红色区域
        mask = extract_red_mask(frame)

        # 检测轮廓
        contours = find_contours(mask)
        contours = filter_contours_by_area(contours, min_area=200, max_area=10000)

        # 创建可视化图像
        vis = frame.copy()
        detected_count = 0

        for contour in contours:
            # 提取 L 型关键点
            result = find_l_shape_keypoints(contour)

            if result is None:
                # 未检测到 L 型，绘制灰色轮廓
                cv2.drawContours(vis, [contour], -1, (128, 128, 128), 1)
                continue

            detected_count += 1

            # 绘制原始轮廓（浅绿色）
            cv2.drawContours(vis, [contour], -1, (0, 200, 0), 1)

            # 绘制凸包（黄色）
            hull = cv2.convexHull(contour)
            cv2.drawContours(vis, [hull], -1, (0, 255, 255), 2)

            # 绘制拟合的两条直线（蓝色）
            for line in result['lines']:
                x1, y1, x2, y2 = line
                cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # 绘制 3 个关键点（红色大圆圈）
            keypoints = result['keypoints']
            labels = ['端点1', '拐点', '端点2']

            for i, (kp, label) in enumerate(zip(keypoints, labels)):
                # 红色实心圆
                cv2.circle(vis, kp, 8, (0, 0, 255), -1)
                # 白色边框
                cv2.circle(vis, kp, 8, (255, 255, 255), 2)
                # 标签
                cv2.putText(vis, label, (kp[0]+12, kp[1]-12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 标注编号（在轮廓中心）
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(vis, f"L#{detected_count}", (cx-20, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        total_detections += detected_count

        # 添加统计信息
        if total_frames > 0:
            info = f"Frame: {frame_count}/{total_frames} | Detected: {detected_count} L-shapes"
        else:
            info = f"Frame: {frame_count} | Detected: {detected_count} L-shapes"
        cv2.putText(vis, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 写入输出视频
        out.write(vis)

        # 每 30 帧打印一次进度
        if frame_count % 30 == 0:
            if total_frames > 0:
                progress = (frame_count / total_frames) * 100
                print(f"进度: {progress:.1f}% ({frame_count}/{total_frames})")
            else:
                print(f"已处理: {frame_count} 帧")

    # 释放资源
    cap.release()
    out.release()

    avg_detections = total_detections / frame_count if frame_count > 0 else 0

    print(f"\n=== 处理完成 ===")
    print(f"总帧数: {frame_count}")
    print(f"总检测数: {total_detections}")
    print(f"平均每帧检测: {avg_detections:.2f} 个 L 型")
    print(f"输出视频已保存到: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 test_video_detection.py <video_path> [output_path]")
        sys.exit(1)

    video_path = sys.argv[1]

    # 如果路径不是绝对路径，添加 data/ 前缀
    if not os.path.isabs(video_path) and not os.path.exists(video_path):
        video_path = os.path.join('data', video_path)

    output_path = sys.argv[2] if len(sys.argv) >= 3 else None

    process_video(video_path, output_path)
