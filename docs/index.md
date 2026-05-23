# Vision 项目文档索引

> RoboMaster 兑换站 L 型灯条识别与位姿估计，基于 OpenCV 链码算法

## 快速导航

- **[MAP.md](../MAP.md)** - 项目架构映射（必读）
- **[开发路线](roadmap.md)** - 当前任务与待办事项
- **[几何模型](geometric-model.md)** - 兑换站尺寸规格

## 模块文档

| 模块 | 功能 | 文档 |
|------|------|------|
| 感知模块 | 红色提取、轮廓检测 | [perception.md](modules/perception.md) |
| 计算模块 | 链码计算、L 型匹配、透视矫正 | [geometric.md](modules/geometric.md) |
| 接口模块 | ROS2 接口（待实现） | [interface.md](modules/interface.md) |

## Agent 协作

- **code-writer**: 算法开发，函数松耦合，可视化验证
- **code-reviewer**: 代码审查，文档更新，进度跟踪

**协作流程**: 先读 MAP.md → 定向读取模块 → 增量更新

## Docker 环境

```bash
# 启动容器
./docker/run.sh

# 运行算法（模块化版本）
docker exec robot_vision bash -c "cd /home/workspace/vision && python3 src/main.py data/1.png"

# 运行透视变换与角点检测测试
docker exec robot_vision bash -c "cd /home/workspace/vision && python3 src/test/test_perspective_transform.py data/1.png"

# 运行旧版本（保留用于对比）
docker exec robot_vision bash -c "cd /home/workspace/vision && python3 src/test/test_chaincode_perspective.py data/1.png"
```

**容器**: robot_vision | **工作目录**: /home/workspace | **镜像**: CUDA 12.1 + ROS2 Humble

## 当前状态

- **阶段**: Python 算法验证
- **检测率**: 55.4% → 目标 >85%
- **主流程**: `src/main.py` 已对齐 `src/test/debug/debug_line_extraction.py`（灰度轮廓红比掩膜、两层候选筛选、六边提线、八链码评分、几何约束）
- **新增抖动边抑制**: 长边若梯度过陡且过多、或整段拟合残差过大，直接判为非近似直线
- **测试数据**:
  - 1.png（正视图，4 个灯条）
  - 2.png（侧视角 45°，5 个灯条：4 正面 + 1 侧面）
  - 3.png（侧视图，1 个侧边灯条）
- **场景**: 最多 1 个静止兑换站，有红色干扰灯条
- **多 L 关联**: k0 / 对径 k1=(k0+3)%6、`corner_k2`；k0 射线 adjacent/opposite 与 CCW 顺序环；`calculation` 包已导出 `analyze_frame_associations`、`QuadDrawStabilizer`、`draw_yellow_quad_and_center` 等（见 `docs/modules/geometric.md`「算法库导出」）
- **read 效果视频**: `src/test/video/test_video_line_extraction.py` → `output/read_effect.mp4`，仅稳定后绘制黄框与框心
