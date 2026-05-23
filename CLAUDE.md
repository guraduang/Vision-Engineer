# RoboMaster 视觉项目 - Claude Code 配置

## 项目概述

RoboMaster 兑换站 L 型灯条识别与位姿估计。基于 OpenCV 链码算法（非深度学习），当前处于 Python 算法验证阶段（检测率 55.4% → 目标 >85%）。

## 递归式访问协议 ⚡

**每次对话必须遵循以下流程**：

1. **先读 [MAP.md](MAP.md)** - 定位修改范围，找到相关模块和行号
2. **定向读取模块文档** - 仅读取 `docs/modules/` 下相关模块的文档
3. **精准读取代码** - 根据 MAP.md 中的行号范围，仅读取受影响的代码段
4. **增量更新** - 仅输出修改的函数，不重写整个文件

**禁止**：一次性读取所有文件、重写整个文件、读取无关模块

## Docker 环境

**容器名**: robot_vision | **工作目录**: /home/workspace | **镜像**: CUDA 12.1 + ROS2 Humble

**注意**: 运行前请确保容器已启动（`docker ps` 检查状态）

```bash
# 启动容器（如果未运行）
./docker/run.sh

# 运行算法（模块化版本）
docker exec robot_vision bash -c "cd /home/workspace/vision && python3 src/main.py data/1.png"

# 运行透视变换与角点检测测试
docker exec robot_vision bash -c "cd /home/workspace/vision && python3 src/test/test_perspective_transform.py data/1.png"

# 运行旧版本（保留用于对比）
docker exec robot_vision bash -c "cd /home/workspace/vision && python3 src/test/test_chaincode_perspective.py data/1.png"
```

## Agent 角色

### code-writer
- 算法开发，先读 MAP.md 定位修改范围
- 仅输出修改的函数，不重写整个文件
- 提供可视化验证

### code-reviewer
- 代码审查，更新 docs/roadmap.md 跟踪进度
- 验证测试结果

## 文档导航

- **[MAP.md](MAP.md)** - 项目架构映射（必读）
- **[docs/index.md](docs/index.md)** - 文档索引
- **[docs/roadmap.md](docs/roadmap.md)** - 开发路线
- **[docs/geometric-model.md](docs/geometric-model.md)** - 几何模型

