# Test 目录结构说明

测试脚本按功能分类存放，保持项目整洁。

## 📁 目录结构

```
test/
├── README.md                    # 本文件
├── agent_helper.py              # Agent 辅助工具
├── agent.sh                     # Agent 启动脚本
│
├── archived/                    # 归档的旧版本测试
│   ├── test_chaincode_perspective.py      # 旧版链码+透视变换
│   ├── test_l_shape_with_perspective.py   # 旧版L型+透视变换
│   ├── test_perspective_transform.py      # 旧版透视变换测试
│   ├── test_config_profiles.py            # 配置文件测试
│   └── visualize_config_effects.py        # 配置效果可视化
│
├── approx_poly/                 # 多边形逼近相关测试
│   ├── test_approx_poly.py                # 多边形逼近测试
│   ├── visualize_approx_poly_angles.py    # 角度梯度可视化
│   ├── test_morphology_smoothing.py       # 形态学平滑测试
│   ├── test_parameter_tuning.py           # 参数调优测试
│   └── test_warped_mask_angle.py          # 变换后掩码角度测试
│
├── corner_detection/            # 角点检测相关测试
│   ├── test_corner_detection.py           # 角点检测测试
│   ├── compare_corner_algorithms.py       # 角点算法对比
│   └── visualize_angle_computation.py     # 角度计算可视化
│
├── debug/                       # 调试工具
│   ├── debug_all_contours.py              # 调试所有轮廓
│   ├── debug_contour_area.py              # 调试轮廓面积
│   └── debug_corner_detection.py          # 调试角点检测
│
└── video/                       # 视频处理测试
    └── test_video_detection.py            # 视频检测测试
```

## 🎯 使用指南

### 当前开发中的测试
- **多边形逼近**: `approx_poly/` - 正在开发的多边形逼近算法
- **角点检测**: `corner_detection/` - 角点检测算法测试

### 调试工具
- **debug/**: 用于调试特定问题的脚本

### 归档文件
- **archived/**: 旧版本测试脚本，保留用于对比和参考

## 🚀 运行测试

所有测试脚本都需要在 Docker 容器中运行：

```bash
# 多边形逼近测试
docker exec robot_vision bash -c "cd /home/workspace/vision && python3 src/test/approx_poly/test_approx_poly.py data/1.png"

# 角点检测测试
docker exec robot_vision bash -c "cd /home/workspace/vision && python3 src/test/corner_detection/test_corner_detection.py data/1.png"

# 视频检测测试
docker exec robot_vision bash -c "cd /home/workspace/vision && python3 src/test/video/test_video_detection.py data/23/2/red.avi"
```

## 📝 添加新测试

1. 根据测试类型选择合适的目录
2. 遵循命名规范：`test_*.py` 或 `visualize_*.py`
3. 在文件开头添加清晰的文档字符串
4. 确保测试可以独立运行
