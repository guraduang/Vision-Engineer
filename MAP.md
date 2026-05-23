# Vision 项目架构映射

> **单一真理来源** - 每次对话先读此文件定位修改范围

## 快速导航

- **[文档索引](docs/index.md)** - 项目文档入口
- **[开发路线](docs/roadmap.md)** - 当前任务与待办事项
- **[几何模型](docs/geometric-model.md)** - 兑换站尺寸规格
- **[Docker 环境](#docker-环境)** - 容器配置与运行命令

## 核心算法（当前：主流程已对齐调试链路）

**主程序**: `src/main.py`

### 感知模块 (src/perception/)

| 模块 | 功能 | 文件 | 关键函数 | 状态 |
|------|------|------|----------|------|
| 红色提取 | R-B 差值法 | `red_extractor.py` | `extract_red_mask()` | ✅ |
| 轮廓检测 | OpenCV findContours | `contour_detector.py` | `find_contours()`, `filter_contours_by_area()` | ✅ |

### 计算模块 (src/calculation/)

| 模块 | 功能 | 文件 | 关键函数 | 状态 |
|------|------|------|----------|------|
| 链码计算 | 8 方向链码 | `chaincode.py` | `get_chain_code()` | ✅ |
| 链码归一化 | 旋转不变性 | `chaincode.py` | `normalize_chain_code()` | ✅ |
| 链码简化 | 合并连续方向 | `chaincode.py` | `simplify_chain_code()` | ✅ |
| L 型匹配 | 六边提线 + 几何约束 + 八链码评分 | `test/debug/debug_line_extraction.py` | `extract_line_segments_from_6_corners()` | ✅ |
| 多 L 关联 | k0→对径射线分类 + CCW 顺序环 + 严格四边形 + 时序 `QuadDrawStabilizer` + 跟踪 | `calculation/l_bar_association.py` | `analyze_frame_associations()`, `classify_l_pair_k0_ray_relation()`, `ordered_quad_passes_k0_ring()`, `QuadDrawStabilizer`, `draw_yellow_quad_and_center` | 🚧 |
| 透视矫正 | 最小旋转矩形 | `perspective.py` | `warp_to_frontal()` | ✅ |

### 接口模块 (src/interface/)

| 模块 | 功能 | 文件 | 状态 |
|------|------|------|------|
| ROS2 接口 | 待实现 | `__init__.py` | 🚧 |

**详细文档**: [感知模块](docs/modules/perception.md) | [计算模块](docs/modules/geometric.md)

## 目标架构（已完成模块化重构）

```
src/
├── perception/              # 感知模块 ✅
│   ├── __init__.py
│   ├── red_extractor.py    # 红色提取
│   └── contour_detector.py # 轮廓检测
├── calculation/             # 计算模块 ✅
│   ├── __init__.py         # 链码 / 透视 / 多 L 关联等算法库导出
│   ├── chaincode.py        # 链码计算与归一化
│   ├── l_shape_matcher.py  # L 型匹配
│   ├── l_bar_association.py # 多 L 关联、k0 射线、顺序环、稳定绘制
│   ├── corner_detector.py   # L 外轮廓三关键点
│   ├── contour_corner_detector.py # 边上角点调试
│   └── perspective.py      # 透视矫正
├── interface/               # 接口模块 🚧
│   └── __init__.py         # ROS2 节点（待实现）
├── main.py                  # 主程序入口（已对齐 debug_line_extraction） ✅
└── test/                    # 测试脚本（已分类整理）
    ├── README.md            # 测试目录说明
    ├── agent_helper.py      # Agent 辅助工具
    ├── archived/            # 归档的旧版本测试
    ├── approx_poly/         # 多边形逼近测试（开发中）
    ├── corner_detection/    # 角点检测测试
    ├── debug/               # 调试工具
    └── video/               # 视频处理测试
```

## Docker 环境

| 组件 | 配置 | 路径 |
|------|------|------|
| 容器名 | robot_vision | - |
| 工作目录 | /home/workspace | 映射到 /home/duang/vision |
| 镜像 | robotics_image:latest | CUDA 12.1 + ROS2 Humble |
| Dockerfile | CUDA + OpenCV + PCL | `docker/Dockerfile` |
| 启动脚本 | GPU 支持 + X11 | `docker/run.sh` |

**运行命令**:
```bash
# 启动容器
./docker/run.sh

# 运行算法（模块化版本）
docker exec robot_vision bash -c "cd /home/workspace/vision && python3 src/main.py data/1.png"

# 运行算法（旧版本，保留用于对比）
docker exec robot_vision bash -c "cd /home/workspace/vision && python3 src/test/test_chaincode_perspective.py data/1.png"
```

## 文档索引

| 文档 | 用途 | 路径 |
|------|------|------|
| 核心索引 | 项目导航 | `docs/index.md` |
| 感知模块 | 红色提取、轮廓检测 | `docs/modules/perception.md` |
| 计算模块 | 链码、L 型匹配、透视矫正 | `docs/modules/geometric.md` |
| 接口模块 | ROS2 接口（待实现） | `docs/modules/interface.md` |
| 开发路线 | 任务与需求变动 | `docs/roadmap.md` |
| 几何模型 | 兑换站尺寸规格 | `docs/geometric-model.md` |

## 关键参数

| 参数 | 默认值 | 说明 | 位置 |
|------|--------|------|------|
| 红色阈值 | 30 | R-B 差值阈值 | `red_extractor.py` |
| 灰度阈值 | 20 | 灰度轮廓提取阈值 | `red_extractor.py` |
| 红色占比阈值 | 0.5 | 轮廓内红色像素占比判定 | `red_extractor.py` |
| 合并阈值 | 3 | 链码简化阈值 | `chaincode.py` |
| 面积范围 | 100-50000 | 第一层面积筛选（主流程） | `contour_detector.py` |
| warped 面积占比 | 0.10-0.585 | 第二层候选筛选（主流程） | `debug_line_extraction.py` |
| min_quad_match_score | 0.30 | 严格四元组链码分下限 | `l_bar_association.py` |
| 算法库入口 | `from calculation import …` | 见 `docs/modules/geometric.md`「算法库导出」 | `calculation/__init__.py` |
| 长边抖动拒绝 | mean>=7.5 且 p90>=14.0 | 长边非直线判定 | `debug_line_extraction.py` |
| 长边陡梯度拒绝 | q95>=40 且 steep_ratio>=0.20 | 高梯度长边拒绝 | `debug_line_extraction.py` |
| 透视变换输出 | 200×200 | 变换后图像尺寸 | `perspective.py` |

## ROS2 模块（C++）

**路径**: `src/realsense_subscriber/`

| 文件 | 功能 |
|------|------|
| `include/realsense_subscriber/parameters.hpp` | 参数管理 |
| `include/realsense_subscriber/realsense_subscriber_node.hpp` | ROS2 节点 |
| `include/realsense_subscriber/vision_pipeline.hpp` | 视觉处理管道 |
| `src/main.cpp` | 主程序入口 |
| `CMakeLists.txt` | 编译配置 |

## Agent 协作

- **code-writer**: 算法开发，先读 MAP.md 定位修改范围，仅输出修改的函数
- **code-reviewer**: 代码审查，更新 roadmap.md 跟踪进度

## 测试数据

| 文件 | 说明 |
|------|------|
| 1.png | 正视图，4 个 L 型灯条（1 个兑换站正面） |
| 2.png | 侧视角约 45°，5 个 L 型灯条（4 个正面 + 1 个侧面） |
| 3.png | 侧视图，1 个侧边灯条 |
| 23/2/red.avi | 视频测试（6711 帧），兑换站静止，部分帧无兑换站，有红色干扰灯条 |
| read.avi（片段） | 由 red.avi **2:00–2:20** 按时间戳裁切的调试片段，用于拥挤/遮挡/侧边灯条干扰回归（`CAP_PROP_POS_FRAMES` 对该编码不可靠时宜用 `POS_MSEC` 顺序读取） |

**场景特点**：
- 场景中最多只有 1 个兑换站（静止不动）
- 视频中部分时间无兑换站出现
- 存在其他红色灯条干扰
- 兑换站不会闪现（静止目标）

## 当前状态

- **阶段**: Python 算法验证
- **检测率**: 55.4% → 目标 >85%
- **优先级**: 提高检测率，增强透视变换鲁棒性
- **暂不考虑**: ROS2 迁移、C++ 重写

---

**递归式访问协议**:
1. 每次对话先读 `MAP.md` 定位修改范围
2. 定向读取相关模块文档（`docs/modules/`）
3. 仅读取受影响的代码行号范围
4. 仅输出修改的函数，不重写整个文件
