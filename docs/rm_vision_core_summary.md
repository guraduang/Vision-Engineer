# rm_vision_core 项目总结

> 华南理工大学华南虎战队 RM2025 能量机关识别算法
> 详细学习笔记：`/home/duang/vision/example/rm_vision_core/STUDY_NOTES.md`

---

## 一、核心价值

### 1.1 创新点
**链码角点提取算法**：将轮廓转化为角度曲线，通过梯度分析提取灯条角点
- 精度：1~3 像素（传统方法 5~8 像素）
- 鲁棒性：抗光晕、拖影干扰
- 通用性：适用于任何条状轮廓的端点提取

### 1.2 技术栈
- C++20 + OpenCV 4.7.0
- 无 ROS 依赖，纯视觉算法
- 模块化设计，易于移植

---

## 二、算法流程

```
图像输入
  ↓
二值化（颜色通道阈值）
  ↓
轮廓提取（findContours）
  ↓
特征识别（5类特征并行提取）
  ├─ 未激活靶心（几何筛选）
  ├─ 已激活靶心（圆形度）
  ├─ 未激活扇叶（矩形匹配）
  ├─ 已激活扇叶（链码角点提取 ★核心）
  └─ 神符中心（R标识别/强制构造）
  ↓
特征配对（靶心+中心+扇叶 → 5个组合体）
  ↓
帧间匹配（组合体 ↔ 追踪器，最小距离）
  ↓
PnP 位姿解算（多角点加权）
  ↓
输出位姿
```

---

## 三、核心算法详解

### 3.1 链码角点提取（已激活扇叶）

**目标**：从扇叶轮廓中提取 6 个精确角点

**步骤**：

1. **轮廓预筛选**
   - 长宽比、面积、填充率快速过滤

2. **链码化 → 角度数组**
   ```cpp
   轮廓点 → 方向向量 → 角度序列
   高斯平滑 → 消除噪声
   角度连续化 → 处理 ±180° 跳变
   ```

3. **梯度计算**
   ```cpp
   角度一阶导数 → 梯度数组
   梯度≈0 的区域 = 直线段
   ```

4. **直线段提取**
   ```cpp
   梯度阈值分割 → 线段列表
   合并相邻线段 → 去除碎片
   ```

5. **正反线段匹配**
   ```cpp
   找角度相差 180° 的线段对
   验证垂直距离（灯条宽度）
   方向判断（叉积）
   ```

6. **凸起检测**
   ```cpp
   顶部凸起 → 3个角点
   底部中心凸起 → 1个角点
   侧面凸起 → 2个角点
   共 6 个角点用于 PnP
   ```

**关键代码**：`modules/feature/rune_fan/src/rune_fan_active.cpp`

---

### 3.2 特征匹配算法

#### 单帧配对（特征 → 组合体）
```cpp
// 将靶心、中心、扇叶配对成 5 个 RuneCombo
for 每个靶心:
    找最近的扇叶（激活状态一致）
    共享同一个神符中心
    → (target, center, fan)
```

#### 帧间匹配（组合体 → 追踪器）
```cpp
// 构建 5×5 代价矩阵（欧氏距离平方）
cost[i][j] = dist²(combo[i], tracker[j])

// DFS 全排列搜索最优匹配（5! = 120 种）
min_cost = ∞
for 每种排列:
    if sum(cost) < min_cost:
        保存此排列

// 更新追踪器
for i in 0..4:
    tracker[i].update(combo[match[i]])
```

**关键代码**：
- `modules/detector/rune_detector/src/rune_detector_find.cpp`
- `modules/detector/rune_detector/src/rune_detector_match.cpp`

---

### 3.3 PnP 位姿解算

```cpp
// 1. 收集所有特征的角点
for (target, center, fan) in combos:
    points_2d += feature.get_image_points()   // 像素坐标
    points_3d += feature.get_world_points()   // 物理坐标(mm)
    weights += feature.get_weights()          // 可信度

// 2. 加权 PnP 求解
solvePnPRansac(points_3d, points_2d, camera_matrix, 
               dist_coeffs, rvec, tvec)

// 3. 输出位姿（旋转向量 + 平移向量）
```

**特点**：
- RANSAC 抗误匹配
- 多角点约束（最多 37 个角点）
- 权重机制（完整特征权重高）

---

## 四、特征层级结构

```
FeatureNode（抽象基类）
├── RuneCenter        神符中心（R标）
├── RuneTarget        靶心
│   ├── Inactive      未激活（灰色）
│   └── Active        已激活（发光）
└── RuneFan           扇叶
    ├── Inactive      未激活
    ├── Active        已激活（6角点）
    └── ActiveIncomplete  残缺（3角点，遮挡情况）

组合层：
RuneCombo = (Target, Center, Fan)  单个神符单元
RuneGroup = 5 × RuneCombo          完整神符
RuneTracker                        帧间追踪器
```

---

## 五、关键文件导航

### 核心算法
```
modules/feature/rune_fan/src/
├── rune_fan_active.cpp          ★ 链码角点提取（核心）
├── rune_fan_inactive.cpp        未激活扇叶识别
└── hump_*.cpp                   凸起检测

modules/detector/rune_detector/src/
├── rune_detector.cpp            主流程入口
├── rune_detector_find.cpp       特征提取与配对
├── rune_detector_match.cpp      ★ 帧间匹配
└── rune_detector_get_pnp_data.cpp  PnP解算
```

### 数据结构
```
common/modules/feature/include/vc/feature/
├── feature_node.h               特征节点基类
└── tracking_feature_node.h      追踪节点

common/contour_proc/include/vc/contour_proc/
└── contour_wrapper.hpp          轮廓包装器

common/math/include/vc/math/
└── pose_node.hpp                位姿节点
```

### 参数配置
```
modules/feature/*/include/vc/feature/
├── rune_fan_param.h             扇叶参数
├── rune_target_param.h          靶心参数
└── rune_center_param.h          中心参数
```

---

## 六、学习路径

### 阶段一：理解数据流（1-2天）
1. 运行 Demo：`examples/rune_detect_demo/main.cpp`
2. 阅读头文件：理解接口定义
3. 单步调试：跟踪一帧的完整处理流程

### 阶段二：掌握核心算法（3-5天）
1. **链码算法**：`rune_fan_active.cpp`
   - 角度数组生成
   - 梯度计算
   - 线段匹配
   
2. **匹配策略**：
   - 单帧配对逻辑
   - 帧间关联算法
   
3. **参数调优**：
   - 修改阈值观察效果
   - 理解参数物理意义

### 阶段三：移植与优化（1-2周）
1. 提取链码算法到自己的项目
2. 适配其他特征（如装甲板灯条）
3. 性能优化（多线程、SIMD）

---

## 七、可借鉴的设计

### 7.1 算法层面
- **链码角点提取**：可用于任何条状轮廓
- **多特征融合**：提高鲁棒性
- **自适应降级**：遮挡时减少角点数量

### 7.2 工程层面
- **模块化设计**：特征独立、易于测试
- **统一接口**：`FeatureNode` 抽象基类
- **参数宏管理**：集中配置、类型安全
- **智能指针**：自动内存管理

### 7.3 调试工具
- 参数可视化管理器
- 日志系统
- 重投影验证

---

## 八、与你的项目对比

### 相同点
- 都是基于轮廓的传统视觉算法
- 都需要角点提取用于位姿估计
- 都面临光照、遮挡等鲁棒性问题

### 差异点
| 维度 | rm_vision_core | 你的项目 |
|------|----------------|----------|
| 目标 | 神符（5扇叶+靶心） | 兑换站 L 型灯条 |
| 特征 | 复杂（凸起结构） | 简单（矩形灯条） |
| 角点数 | 最多 37 个 | 8 个（4个L型） |
| 算法 | 链码角点提取 | 链码 L 型匹配 |
| 语言 | C++ | Python（验证阶段） |

### 可借鉴的部分
1. **链码算法思路**：角度曲线 → 梯度 → 线段匹配
2. **多特征融合**：L型 + 透视变换 + 几何约束
3. **帧间追踪**：最小距离匹配 + 卡尔曼滤波
4. **参数管理**：集中配置、可视化调试

---

## 九、关键收获

### 算法思想
1. **轮廓 → 角度曲线**：降维表示，便于分析
2. **梯度分析**：找直线段（梯度≈0）
3. **正反线段对**：利用对称性定位灯条
4. **多层筛选**：快速几何过滤 + 精细角点提取

### 工程实践
1. **模块化**：每个特征独立实现
2. **统一接口**：多态设计，易于扩展
3. **参数化**：阈值可调，便于优化
4. **可视化**：每个阶段都有调试输出

---

## 十、下一步行动

### 对于你的项目
1. **借鉴链码思路**：优化 L 型角点提取精度
2. **引入帧间追踪**：提高检测稳定性
3. **多特征融合**：L型 + 透视变换 + 几何模型
4. **参数可视化**：建立调试工具

### 深入学习
1. 阅读完整源码：`rm_vision_core/modules/`
2. 运行并修改参数：观察效果变化
3. 移植核心算法：到你的 Python 项目
4. 性能对比：链码 vs 传统方法

---

**完整学习笔记**：`/home/duang/vision/example/rm_vision_core/STUDY_NOTES.md`（360行详细注释）
