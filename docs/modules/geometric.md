# 计算模块

> 链码计算、L 型匹配、透视矫正

## 输入/输出协议

### get_chain_code(contour: np.ndarray) -> List[int]
- **输入**: 轮廓点集
- **输出**: 8 方向链码 [0-7]
- **方向**: 0=右, 2=上, 4=左, 6=下
- **位置**: `src/calculation/chaincode.py`

### normalize_chain_code(chain: List[int]) -> List[int]
- **输入**: 原始链码
- **输出**: 归一化链码（旋转不变性）
- **方法**: 字典序最小的旋转
- **位置**: `src/calculation/chaincode.py`

### simplify_chain_code(chain: List[int], merge_threshold: int = 3) -> List[Tuple[int, int]]
- **输入**: 链码，合并阈值
- **输出**: [(direction, count), ...]
- **方法**: 合并连续相同方向
- **位置**: `src/calculation/chaincode.py`

### warp_to_frontal(image: np.ndarray, contour: np.ndarray, dst_size: int = 200, padding: int = 10) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]
- **输入**: 原图 + 轮廓 + 输出边长 + 四周留白
- **输出**: (透视矫正后的图像, 矫正后的轮廓)
- **方法**:
  1. `cv2.minAreaRect(contour)` 得到最小旋转外接矩形
  2. `cv2.boxPoints` 取四角，排序为 TL→TR→BR→BL 后与目标矩形四角一一对应
  3. `getPerspectiveTransform` + `warpPerspective` 到 `dst_size×dst_size`，有效区域为内侧 `(dst_size - 2*padding)²`
- **默认输出**: 200×200 像素（与 `perspective.py` 一致）
- **位置**: `src/calculation/perspective.py`

### is_l_shape_chaincode(contour: np.ndarray, image: np.ndarray, epsilon_factor: float = 0.02) -> Tuple[bool, np.ndarray, Dict]
- **输入**: 轮廓, 原图, 轮廓近似因子
- **输出**: (是否为 L 型, 近似轮廓, 详细信息字典)
- **判断标准**: 主方向占比 ≥50%，段数 4-8，边对匹配 ≥2
- **位置**: `src/calculation/l_shape_matcher.py`

### get_all_segment_corners(contour: np.ndarray, image_size: int = 300) -> Optional[Dict]
- **输入**: 轮廓、参考边长（通常取 warped 图 `max(h,w)`）
- **输出**: 单调区间端点经聚类后的**全部**角点索引与坐标、`segments`、`angles`、`gradient`；不可用时返回 `None`
- **位置**: `src/calculation/contour_corner_detector.py`
- **用途**: 调试链码提线时查看边上全部过渡角点（不限于 top-3 外角点）

### extract_line_segments_from_6_corners(contour: np.ndarray, ...) -> Dict
- **输入**: 轮廓（通常为 warped 图最大红轮廓）
- **输出**: 六边提线结果（`ok`, `lines`, `chain8`, `lengths8`, `enclosing_triangle`, `segment_debug` 等）
- **方法**: `approxPolyDP(6角)` -> 角点分段 -> 梯度引导拟合 -> 长边抖动/陡梯度拒绝；同时计算最小外接三角形供单 L 外肘与两臂射线方向估计
- **位置**: `src/test/debug/debug_line_extraction.py`
- **主流程**: `src/main.py` 直接复用该函数

### build_lbar_observation_from_six_corners(candidate_idx, contour, corner_driven, proc_src, warp_m, match_score=-1.0, template_variant=None) -> Optional[LBarObservation]
- **输入**: L 型候选编号、原图轮廓、六角点提线结果、处理空间标记、原图到 warped 的透视矩阵；`match_score` / `template_variant` 保留兼容，当前实现忽略
- **输出**: 原图坐标下的 `LBarObservation`：`corner_k`（k0 外拐点拟合交点细化）、`corner_k2`（对径角 **k1=(k0+3)%6** 的拟合交点细化，原图）、`dir_a`/`dir_b`、`bisector`、`centroid`、`arm_a_end`/`arm_b_end`、`match_score`（常为 -1）
- **方法**: **k0** 仍为「外凸可角 + 邻边 `lengths8` 和最大」顶点；**k1** 固定为六角闭合上 **对径索引** `(k0+3)%6`。K 与 k1 均由相邻两边拟合线求交并相对六角顶点门控（见 `_refined_corner_proc_at_vertex`）。`proc_src=='warped'` 时用逆透视将 K、k1 映回原图。两臂为沿拟合边朝外的单位方向；**仅**角平分线与质心方向相反时翻转 `bisector`，不翻转两臂
- **位置**: `src/test/debug/debug_line_extraction.py`
- **用途**: 与 `k0_to_opposite_corner_ray` / `classify_l_pair_k0_ray_relation` 一致的几何语义；视频 `test_video_line_extraction.py` 用其构造观测

### weighted_chaincode_match(chain, lengths, template=[0,6,4,6,4,2]) -> Tuple
- **输入**: 六边链码与边长
- **输出**: 最佳加权匹配分数、模板位移、稳定化链码
- **方法**: 最长4边高权重 + 模板循环位移 + 反向模板匹配
- **位置**: `src/test/debug/debug_line_extraction.py`
- **主流程**: `src/main.py` 直接复用该函数

### find_l_shape_keypoints(contour: np.ndarray) -> Dict
- **输入**: L 型轮廓
- **输出**: 包含 3 个外轮廓角点的字典 [端点1, 拐点, 端点2]
- **方法**: 基于最小外接矩形的 3 个顶点，识别拐点（夹角最接近 90°）；若该拐点明显偏离凸包，则在贴近凸包的候选点中重选，降低内凹点误用概率
- **位置**: `src/calculation/corner_detector.py`

### find_inner_corners_from_outer(outer_keypoints, all_corners, warped_image) -> Optional[Tuple[int, int]]
- **输入**: 外轮廓 3 个角点, 所有检测到的角点, 变换后的图像
- **输出**: 内拐点坐标，找不到返回 None
- **方法**: Shi-Tomasi 角点检测 + 多重过滤条件
- **过滤条件**:
  - 不是外轮廓的 3 个角点（距离 > 30 像素）
  - **必须在红色区域内**（通过红色掩码验证）
  - 距离外拐点在合理范围（外轮廓边长的 25%-75%）
  - 在外拐点到三角形中心的方向上（方向一致性 > 0.5）
- **位置**: `src/test/test_perspective_transform.py`

## 关键参数

| 参数 | 默认值 | 说明 | 行号 |
|------|--------|------|------|
| merge_threshold | 3 | 链码简化阈值 | L68 |
| main_direction_ratio | 0.5 | 主方向占比阈值 | 代码中 |
| segment_count_range | (6, 10) | 简化链码段数范围 | 代码中 |
| dst_size | 200 | 透视变换输出边长 | `perspective.py` |
| padding | 10 | 透视变换四周留白像素 | `perspective.py` |
| quality_level | 0.05 | Shi-Tomasi 角点质量阈值 | 代码中 |
| min_distance | 30 | 角点之间最小距离 | 代码中 |
| inner_corner_distance_range | (0.25, 0.75) | 内拐点距离范围（相对外轮廓边长） | 代码中 |
| gradient_smooth_window | 5 | 角度梯度的闭合滑动平均窗口（抑制锯齿抖动） | `debug_line_extraction.py` |
| bridge_max_gap | 2 | 直线 mask 允许桥接的小断裂长度（减少接缝切分） | `debug_line_extraction.py` |
| long_edge_min_points | 45 | 长边判定最小点数 | `debug_line_extraction.py` |
| steep_grad_abs | 30.0 | 陡梯度绝对阈值 | `debug_line_extraction.py` |
| steep_grad_q95 | 40.0 | 长边陡梯度 q95 阈值 | `debug_line_extraction.py` |
| steep_ratio_max | 0.20 | 长边陡梯度占比阈值 | `debug_line_extraction.py` |
| curve_mean_dist_max | 7.5 | 长边整段拟合平均残差阈值 | `debug_line_extraction.py` |
| curve_p90_dist_max | 14.0 | 长边整段拟合 p90 残差阈值 | `debug_line_extraction.py` |
| min_quad_match_score | 0.30 | 严格四元组要求每个观测链码分 ≥ 此值（`-1` 不参与门控） | `l_bar_association.py` |
| k0 射线成对加权底限 | 0.58 | `classify_l_pair_k0_ray_relation` 判为 adjacent/opposite 时对 `adj`/`opp` 的 `max` 抬升 | `l_bar_association.py` |
| QuadDrawStabilizer | 见下 | 视频稳定显示默认：连续 12 帧、跳变 ≤32px、丢帧 grace 6、跳变 break grace 4、EMA α=0.38 | `test_video_line_extraction.py` |

## 算法库导出（`from calculation import …`）

`src/calculation/__init__.py` 对外 re-export 多 L 关联与绘制相关符号，便于脚本与外部节点统一入口：

| 符号 | 说明 |
|------|------|
| `LBarObservation` / `AssociationFrameResult` | 单帧观测与关联结果数据类 |
| `observation_from_contour` | 链码角点路径构造观测（无 `corner_k2` 时不参与 k0 射线分类） |
| `analyze_frame_associations` | 主入口：成对图 + 严格四边形 / fallback + k0 顺序环门控 |
| `estimate_frame_center_from_four` | 2/3/4+ inlier 框心估计 |
| `k0_to_opposite_corner_ray` | 射线 (原点, 单位方向)：k0→对径角 |
| `classify_l_pair_k0_ray_relation` | `'adjacent'` \| `'opposite'` \| `None` |
| `ordered_quad_passes_k0_ring` | 四枚 L CCW 顺序环：边=相交近垂直，对角=平行反向（无 `corner_k2` 时跳过校验） |
| `MultiLBarTracker` / `get_default_l_bar_tracker` / `reset_default_l_bar_tracker` | 质心跟踪 + `observation_stability_mask` |
| `QuadDrawStabilizer` | 四边形+框心时序锁定、EMA、丢帧/跳变 grace |
| `draw_yellow_quad_and_center` | 仅黄框 + 框心十字（视频极简叠加） |
| `draw_association_debug` | 完整调试叠加（四边形、K、臂、射线、成对连线等） |

透视：`warp_to_frontal`、`warp_to_frontal_with_matrix` 亦从 `calculation` 包导出（与视频/主流程候选筛选一致）。

## 算法流程

### 链码计算
1. 遍历轮廓点
2. 计算相邻点的方向角
3. 映射到 8 方向 (0-7)

### 链码归一化
1. 尝试所有旋转起点
2. 找到字典序最小的旋转
3. 实现旋转不变性

### 链码简化
1. 合并连续相同方向
2. 过滤短段（<merge_threshold）
3. 输出 (方向, 长度) 对

### L 型匹配
1. 计算主方向 (0, 2, 4, 6) 占比
2. 检查简化链码段数 (6-10)
3. 判断是否符合 L 型特征

### 透视矫正
1. 计算最小旋转矩形
2. 对 4 个角点排序为 TL→TR→BR→BL，与目标矩形同序对应
3. 透视变换到正面视角（默认 200×200，内侧保留 padding=10）
4. 通过变换矩阵映射外轮廓角点

### 内拐点检测
1. 在变换后的图像中使用 Shi-Tomasi 角点检测
2. 提取红色掩码，确保角点在红色区域内
3. 过滤候选点：
   - 排除外轮廓的 3 个角点
   - 必须在红色区域内
   - 距离外拐点在合理范围（25%-75% 外轮廓边长）
   - 在外拐点到三角形中心的方向上
4. 综合评分选择最佳内拐点（方向一致性 50% + 距离三角形中心 50%）

## L 型链码特征

标准 L 型链码: `[0, 6, 4, 6, 4, 2]` (6 条边)

**特征**:
- 主方向 (0, 2, 4, 6) 占比高
- 段数适中 (6-10)
- 方向变化规律

## 当前问题

- 侧视角（2.png）透视变形，链码可能失真
- 侧视图（3.png）仅 1 个侧边灯条，链码计算不准确
- 场景中有红色干扰灯条，需更鲁棒的 L 型判断标准
- 视频中兑换站静止，可利用时序信息提高稳定性

## 多 L 关联（`l_bar_association.py`）

- **坐标**：优先从六角点提线结果派生外拐点 K（k0）与对径点 `corner_k2`（k0+3），并逆透视回原图；不再在关联阶段重新选 keypoint。
- **观测**：`LBarObservation` 含 `corner_k`、`corner_k2`（可选）、外向 `dir_a`/`dir_b`、`bisector`、`match_score`、`arm_a_end`/`arm_b_end`。无 `corner_k2` 时（如 `observation_from_contour`）跳过 k0 射线顺序环与射线分类抬权。
- **成对**：`_ray_relation_consistency` = max(`_adjacent_consistency`, `_bisector_ray_consistency`)；对顶 `_opposite_consistency`。`match_score` 经 `_pair_quality` 调制。另对每对调用 **`classify_l_pair_k0_ray_relation`**：k0→`corner_k2` 射线 + 两臂匹配；**adjacent**（直线相交近 90° + 一平行一反平行臂）、**opposite**（射线近似反向 + 双臂均反平行）；命中则将对应 `adj` 或 `opp` 抬至不低于 **0.58**。
- **顺序环（抑制误聚类）**：四元组须通过 **`ordered_quad_passes_k0_ring`**：四枚 K 按 CCW 排序后，**环上四边**各满足 `k0_ordered_edge_ok`（分类 adjacent 或几何相交近垂直 + 臂模式），**两对角**各满足 `k0_ordered_diagonal_ok`（分类 opposite 或平行反向 + 双臂反平行）。枚举严格四边形时附加此门控；fallback 若 inlier>4 则 **`_prune_inliers_to_best_ordered_quad`** 选最高分合法四元，恰 4 但不通过环则清空 inlier。
- **主筛选（四边形）**：枚举四个观测的 K 点，`match_score` 门控、严格凸四边形、`_quad_passes_exchange_shape`、强 comb + 强 adjacent/strong opposite 计数、帧中心在形内、共识聚类；额外观测落入四边形且与角强关联则整帧放弃严格结果。
- **逆透视射线约束（`debug_line_extraction`）**：warped 候选逆变换线段与原图两臂/角中位线一致性（主流程/调试用）。
- **回退**：严格失败时并查集 + 上述剪枝/环校验；中心须过 `_center_inside_expanded_k_bounds`。
- **框中心**：`estimate_frame_center_from_four`：2/3/4+ 分档（含对角线交点与 `minAreaRect` 融合）。
- **跟踪**：`MultiLBarTracker` + `observation_stability_mask`。
- **调试图**：`draw_association_debug` 中青绿色闭合框、K、臂、k0 绿色射线、邻边/对角 K–K 线、四 inlier 时 CCW 序号 0–3 等；**视频极简**用 **`draw_yellow_quad_and_center`**（黄框 + 框心），四边形经 **`QuadDrawStabilizer`**（连续一致帧上屏、EMA、丢帧/跳变 grace）后再绘制，见 `src/test/video/test_video_line_extraction.py`。

## 优化方向

- 亚像素角点优化
- 多尺度链码匹配
- 基于几何约束的 L 型判断（角度、边长比例）
- 时序平滑（卡尔曼滤波、移动平均）；多 L 层已提供简易 EMA 跟踪入口

## 调试链路（直线→八链码）

用于 `src/test/debug/debug_line_extraction.py` 的最新策略：

1. 透视后提取候选轮廓线段（最多 6 条）
2. 对 `|梯度|` 做闭合滑动平均，结合 Top-K 峰值自适应阈值确定直线带
3. 在闭合序列上桥接短缺口（小断裂），降低同一长边被噪声切分的概率
4. 相邻线段合并（含闭合轮廓首尾补充合并，gap 采用真实间隔点数）
5. 自动比较 `roll_seam_to_corner=False/True` 两套提线结果，优先选择更接近 6 段且短噪声更少的结果
6. 线段角度映射四方向链码（0/2/4/6）
7. 匹配前进行链码稳态化：合并相邻同向段并吞并短噪声段
8. 以模板 `[0, 6, 4, 6, 4, 2]` 做循环位移 + 反向模板匹配
9. 最长 4 条线加权（权重更高），输出加权匹配分数

## 主流程对齐状态

- `src/main.py` 已对齐上述链路：候选筛选 -> 六边提线 -> 几何约束 -> 八链码评分。
- 接收判据采用 `validate_l_shape_edges(lines)` 的几何约束结果，链码分数用于辅助观测与排序。
