# rm_vision_core 学习笔记
> 华南理工大学华南虎战队 RM2025 能量机关自瞄算法
> 重点：特征提取与匹配

---

## 一、项目概览

这是一套针对 **能量机关（神符）多角点识别与自瞄** 的 C++ 视觉算法框架。

**核心创新**：基于灯条"骨架"的角点提取算法，精度 1~3 像素，远优于传统多边形近似（误差 2~8 像素）。

**技术栈**：C++20 + OpenCV 4.7.0 + Eigen3 + Ceres Solver 3

---

## 二、整体数据流

```
原始图像
   ↓ binary()           二值化（按颜色通道阈值）
   ↓ findContours()     提取所有轮廓
   ↓ extractRuneFeatures()  从轮廓中识别各类特征
        ├── RuneTarget::find_inactive_targets()  未激活靶心
        ├── RuneTarget::find_active_targets()    已激活靶心
        ├── RuneFan::find_inactive_fans()        未激活扇叶
        ├── RuneFan::find_active_fans()          已激活扇叶
        └── RuneCenter::find()                  神符中心(R标)
   ↓ filter*()          多轮筛选，去除噪声候选
   ↓ getMatchedFeature() 将靶心+中心+扇叶配对成组合体
   ↓ getPnpData()        PnP 位姿解算
   ↓ match()            与追踪器匹配（帧间关联）
   ↓ 输出位姿
```

---

## 三、特征层级结构

神符由以下特征组成，采用**逐层组装**设计：

```
FeatureNode（抽象基类）
├── RuneCenter        神符中心（R标，旋转轴心）
├── RuneTarget        神符靶心（待击打的圆形目标）
│   ├── RuneTargetInactive  未激活靶心（灰色）
│   └── RuneTargetActive    已激活靶心（发光）
└── RuneFan           神符扇叶（靶心周围的扇形图案）
    ├── RuneFanInactive     未激活扇叶
    ├── RuneFanActive       已激活扇叶（有凸起结构）
    └── RuneFanActiveIncomplete  残缺已激活扇叶（遮挡情况）

组合层：
RuneCombo = (RuneTarget, RuneCenter, RuneFan)  一个完整神符单元
RuneGroup = 5个 RuneCombo                      整个神符（5扇叶）
RuneTracker                                    帧间追踪单元
```

所有特征节点通过 `FeatureNode_ptr`（`shared_ptr<FeatureNode>`）统一接口连接。

---

## 四、【重点】特征提取算法

### 4.1 已激活扇叶角点提取（核心算法）

**文件**：`modules/feature/rune_fan/src/rune_fan_active.cpp`

已激活扇叶由 5 个长方形灯条组成，算法目标是提取它们相交处的 6 个角点。

#### 步骤一：轮廓预筛选 `make_feature()`

```cpp
// 用旋转矩形做快速几何筛选，排除明显不符合的轮廓
double side_ratio = width / height;
if (side_ratio > ACTIVE_MAX_SIDE_RATIO) return nullptr;  // 长宽比过大

double area_ratio = area / rect_area;
if (area_ratio > MAX || area_ratio < MIN) return nullptr; // 填充率异常

double area_perimeter_ratio = area / (perimeter * perimeter);
if (...) return nullptr;  // 面积周长比（形状复杂度）
```

**关键思路**：用面积、长宽比、填充率三个几何特征快速过滤，避免对每个轮廓都跑重量级算法。

#### 步骤二：链码化 → 角度数组 `getAngles()`

```cpp
// 1. 将轮廓点存入 2×N 的 Mat（x行, y行）
Mat contours_mat(2, contour_plus.size(), CV_32F);

// 2. 用差分核 [-1, 1] 计算相邻点的方向向量（即链码方向）
Mat direction_kernel = (Mat_<float>(1, 2) << -1, 1);
filter2D(contours_mat, directions_mat, -1, direction_kernel, ...);

// 3. 高斯滤波平滑方向，消除噪声
Mat kernel = getGaussianKernel(filter_len, sigma, CV_32F).t();
filter2D(directions_mat, directions_mat, -1, kernel, ...);

// 4. 将方向向量转换为角度（atan2），并处理跨越 ±180° 的连续性
float angle = rad2deg(atan2(*p_y++, *p_x++));
// n 用于累计圈数，保证角度连续（不跳变）
if (angle - last_angle < -180) n++;
else if (angle - last_angle > 180) n--;
*angle_p++ = angle + n * 360;
```

**关键思路**：
- 轮廓点序列 → 方向角序列，把"形状"转化为"角度曲线"
- 高斯平滑消除像素级噪声，保留宏观方向变化
- 角度连续化处理，避免 +180°/-180° 跳变干扰后续梯度计算

#### 步骤三：梯度计算 `getGradient()`

```cpp
// 用 [-1, 0, 1] 核计算角度的一阶导数（梯度）
Mat kernel = (Mat_<float>(1, 3) << -1, 0, 1);
filter2D(angles_mat, gradient_mat, -1, kernel, ...);
```

**关键思路**：梯度接近 0 的区域 = 角度变化平缓 = 轮廓沿直线走 = 灯条的直边部分。

#### 步骤四：提取直线段 `getAllLine()`

```cpp
// 梯度绝对值 <= 3 的连续区间 → 一条直线段
if (abs(grad) <= 3) {
    // 记录线段的起止索引和角度
}

// 合并相邻且角度相近的短线段（间距<20，角度差<5°）
if (next_s - prev_e < 20 && abs(next_ea - prev_sa) < 5) {
    // 合并为一条线段
}
```

**输出**：`Line` 结构体列表，每条线段包含：起止索引、平均角度、中心点。

#### 步骤五：正反线段匹配 `matchLine()`

```cpp
// 寻找角度相差约 180° 的线段对（灯条的两条平行边）
float delta_angle = abs(lines[j].angle - lines[i].angle);
if (abs(delta_angle - 180) > max_angle_delta) continue;  // 角度差不接近180°则跳过

// 计算两线段的垂直距离（即灯条宽度）
float v_dist = getLineVerticalDistance(lines[i], lines[j]);
if (v_dist > max_vertical_distance) continue;  // 距离过远则跳过

// 方向判断：确保 up 在 down 的正确一侧
Point2f dir = Point2f(cos(deg2rad(up.angle)), sin(deg2rad(up.angle)));
return dir.cross(down.center - up.center) > 0;  // 叉积判断方向
```

**关键思路**：一根灯条的两条长边，在角度曲线上表现为两段角度相差 180° 的平行线段。通过匹配这样的"正反线段对"来定位每根灯条。

#### 步骤六：线段对合并与角度矫正

```cpp
// mergeLinePairs(): 合并重叠/重复的线段对
// 判据：中心距 < 20px，角度差 < 10°，长度比 < 3

// correctLineAngle(): 利用正反线段的对称性矫正角度
// 正反线段的平均角度 = 灯条方向，两者应各偏 ±90°
float ave_angle = (up_line.angle + down_line.angle) / 2.0f;
up_line.angle = up_line.angle > ave_angle ? ave_angle + 90 : ave_angle - 90;
```

#### 步骤七：凸起检测 `getActiveFunCorners()`

```cpp
// 获取线段对后，分别检测四类凸起：
auto top_humps = TopHump::getTopHumps(...);           // 顶部凸起（3个角点）
auto bottom_center_humps = BottomCenterHump::get...(); // 底部中心凸起（1个角点）
auto side_humps = SideHump::getSideHumps(...);         // 侧面凸起（2个角点）
// 共 6 个角点，用于 PnP 解算
```

---

### 4.2 未激活扇叶识别

**文件**：`modules/feature/rune_fan/src/rune_fan_inactive.cpp`

未激活扇叶形状更规则（类矩形），使用传统几何筛选：
- 旋转矩形长宽比
- 面积范围
- 与靶心的相对位置关系（方向箭头指向靶心）

---

### 4.3 靶心识别

**文件**：`modules/feature/rune_target/src/`

- **未激活靶心**：同心圆结构，通过轮廓层级（hierarchy）判断是否有子轮廓
- **已激活靶心**：发光的圆形，面积和圆形度筛选

---

### 4.4 神符中心（R标）识别

**文件**：`modules/feature/rune_center/src/rune_center.cpp`

R标是神符的旋转轴心，识别方式：
1. 直接从轮廓中识别（圆形度高、面积适中）
2. 强制构造（当R标被遮挡时）：
   - 方法A：通过未激活扇叶的箭头方向延长线交点推算
   - 方法B：对3个以上靶心做外接圆拟合，圆心即为旋转中心
   - 方法C：对2个以上扇叶的方向向量求交点

```cpp
// 方法B：靶心外接圆拟合
auto [center, radius] = contour_temp->fittedCircle();
targets_center = std::make_unique<Point2f>(center.x, center.y);

// 方法C：扇叶方向线交点
Vec4f line1 = Vec4f(v1.x, v1.y, p1.x, p1.y);
Vec4f line2 = Vec4f(v2.x, v2.y, p2.x, p2.y);
Point2f intersection = getLineIntersection(line1, line2);
```

---

## 五、【重点】特征匹配算法

### 5.1 特征组合配对 `getMatchedFeature()`

**文件**：`modules/detector/rune_detector/src/rune_detector_find.cpp`

将识别到的靶心、中心、扇叶配对成 5 个 `RuneFeatureCombo = (target, center, fan)`。

配对规则：
- 每个靶心对应一个扇叶（激活状态必须一致）
- 所有组合体共享同一个神符中心
- 通过角度/距离关系确定靶心与扇叶的对应关系

### 5.2 帧间追踪匹配 `match()`

**文件**：`modules/detector/rune_detector/src/rune_detector_match.cpp`

将当前帧识别到的 5 个组合体与上一帧的 5 个追踪器进行关联，解决"哪个是哪个"的问题。

```cpp
// 构建 5×5 代价矩阵（欧氏距离平方）
for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 5; j++)
        cost_matrix[i][j] = pow(getDist(combo_center[i], tracker_center[j]), 2);

// DFS 枚举所有排列，找最小总代价（5! = 120种，可接受）
function<void(int, float)> dfs = [&](int u, float sum_cost) {
    if (u == 5) {
        if (sum_cost < min_cost) { min_cost = sum_cost; match = current_match; }
        return;
    }
    for (int i = 0; i < 5; i++) {
        if (!combo_vis[i]) {
            combo_vis[i] = true;
            current_match[u] = i;
            dfs(u + 1, sum_cost + cost_matrix[u][i]);
            combo_vis[i] = false;
        }
    }
};

// 动态计算最大允许偏移（基于追踪器分布的最小外接矩形）
RotatedRect rect = minAreaRect(tracker_centers);
float max_offset = min(rect.size.width, rect.size.height) * 0.3f;
```

**关键思路**：
- 用**匈牙利算法思想**（这里用 DFS 暴力枚举，因为 n=5 很小）
- 代价 = 位置距离平方，最小化总代价 = 最优匹配
- 动态阈值：根据神符实际大小自适应设置最大偏移

### 5.3 掉帧处理

```cpp
if (is_vanish_update) {
    rune_tracker->updateVisible(false);  // 标记为不可见，但保留追踪器
} else {
    rune_tracker->updateVisible(true);
}
```

当某帧识别失败时，追踪器不销毁，等待下一帧重新关联。

---

## 六、PnP 位姿解算

**文件**：`modules/detector/rune_detector/src/rune_detector_get_pnp_data.cpp`

每个特征提供 `getPnpPoints()` 接口，返回：
- `points_2d`：图像坐标系下的角点像素坐标
- `points_3d`：特征坐标系下的角点真实3D坐标（从配置文件读取）
- `weights`：各角点的权重（可信度）

```cpp
// 已激活扇叶的 PnP 点（最多 8 个角点）
if (isSetTopHumpCorners())          // 顶部3个角点
if (isSetBottomCenterHumpCorners()) // 底部中心1个角点
if (isSetSideHumpCorners())         // 侧面2个角点
if (isSetBottomSideHumpCorners())   // 底侧2个角点
```

多特征联合 PnP：将所有可见特征的 2D/3D 点合并，一次求解整个神符的位姿，角点越多精度越高。

---

## 七、关键设计模式

### 7.1 特征节点统一接口

所有特征继承 `FeatureNode`，通过 `FeatureNode_ptr` 传递，需要具体类型时用 `dynamic_pointer_cast`：

```cpp
auto fan = RuneFan::cast(p_feature);  // 等价于 dynamic_pointer_cast<RuneFan>
```

### 7.2 DEFINE_PROPERTY 宏

项目用自定义宏自动生成 getter/setter/isSet：

```cpp
DEFINE_PROPERTY(ActiveFlag, public, protected, (bool));
// 自动生成：
// bool getActiveFlag() const;
// void setActiveFlag(bool);
// bool isSetActiveFlag() const;
```

### 7.3 轮廓封装 ContourWrapper

对 OpenCV 原始轮廓的封装，缓存常用计算结果（面积、周长、外接矩形、拟合椭圆等），避免重复计算：

```cpp
auto area = contour->area();          // 缓存
auto rect = contour->minAreaRect();   // 缓存
auto [center, radius] = contour->fittedCircle();  // 缓存
```

---

## 八、与你的 L 型灯条项目的对比

| 对比项 | rm_vision_core（神符） | 你的项目（L型灯条） |
|--------|----------------------|-------------------|
| 目标形状 | 扇叶（5个长方形组合） | L型（2个长方形组合） |
| 角点提取 | 链码→角度曲线→梯度→线段对 | 链码方向分析 |
| 匹配方式 | 几何关系+帧间追踪 | 待开发 |
| 位姿解算 | 多特征联合PnP | 单特征PnP |
| 抗遮挡 | 残缺扇叶降级识别 | 待开发 |

**可借鉴的核心思路**：
1. **链码→角度曲线→梯度→线段提取** 这套流程完全适用于 L 型灯条
2. **正反线段匹配**（角度差约180°）可用于识别 L 型的两条边
3. **几何筛选三件套**（长宽比+面积+填充率）是通用的快速过滤方法
4. **强制构造中心**的思路（当关键特征缺失时用几何关系推算）值得参考
