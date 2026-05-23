# 开发路线

> 当前任务、待办事项、需求变动记录

## 当前阶段：Python 算法验证

**目标**: 提高检测率（55.4% → >85%），增强透视变换鲁棒性

**场景约束**：
- 最多 1 个静止兑换站
- 视频中部分帧无兑换站
- 存在红色干扰灯条
- 兑换站不会闪现（静止目标）

### 已完成
- ✅ **算法库导出（2026-04-26）**：`src/calculation/__init__.py` re-export 多 L 关联、射线分类、顺序环、`QuadDrawStabilizer`、黄框/调试绘制等；`docs/modules/geometric.md` 增补「算法库导出」表与 k0 顺序环 / 稳定器说明
- ✅ **多 L 观测与关联稳健化（2026-04-26）**：六角点拟合边交点细化外拐点 K；链码 `match_score` 参与成对权重；角平分线双向试探与严格四边形 `min_quad_match_score` 门控；取消两点 fallback 中心 + K 包络扩张 bounds 校验；调试图区分拟合臂线与 K→中心辅助线；在 red.avi 2:00–2:20 裁切（read）上显著压低误报中心帧，静态图 1/2/3 回归通过
- ✅ 红色提取切换为“灰度轮廓 + 轮廓内红色占比”方案
- ✅ L 型外轮廓角点提取（3 个角点：端点1、拐点、端点2）
- ✅ 透视变换（minAreaRect + 角点排序，200×200 输出，padding=10）
- ✅ 内拐点检测（Shi-Tomasi + 红色区域验证）
- ✅ 调试链路两层候选筛选（第一层几何筛选，第二层透视后面积占比）
- ✅ 直线段提取增强（自适应梯度阈值 + 相邻合并 + 闭合首尾补充合并 + 最长6条保留）
- ✅ 6线角度转四链码并实现加权模板匹配（最长4段高权重）
- ✅ **视频时序稳定（2026-04-26）**：`QuadDrawStabilizer`（锁定 + EMA + 丢帧/跳变 grace）+ `draw_yellow_quad_and_center`，read 效果管线仅稳定后绘框与框心

### 进行中
- [ ] 提升多 L 关联召回（保持 2.png 抑制误检的同时，提高视频中心点覆盖率）

### 待办
- [ ] 完善可视化工具（每个阶段独立可视化）
- [ ] 添加单元测试（pytest）
- [ ] 性能分析与优化
- [ ] 视频处理扩展（多序列、ROS 节点侧复用稳定器）

## 已完成阶段：模块化重构

**目标**: 将单文件拆分为模块化结构 ✅

### 已完成
- ✅ 创建 `src/perception/` 模块（red_extractor.py, contour_detector.py）
- ✅ 创建 `src/calculation/` 模块（chaincode.py, l_shape_matcher.py, perspective.py, corner_detector.py）
- ✅ 创建 `src/interface/` 模块（待实现 ROS2 接口）
- ✅ 创建主程序入口 `src/main.py`
- ✅ 更新文档（MAP.md, 模块文档）

### 待完善
- [ ] 编写模块单元测试（pytest）
- [ ] 完善 API 文档和使用示例

## 未来阶段：ROS2 迁移

**目标**: 集成到 ROS2 系统

### 待办
- [ ] 定义自定义消息类型
- [ ] 实现 Python ROS2 节点
- [ ] 集成链码检测算法
- [ ] 添加 PnP 位姿估计
- [ ] 性能优化（实时性 >30 FPS）
- [ ] 部署到 RK3588（NPU 加速）

## 需求变动记录

### 2026-04-16
- **角点提取完成**: 实现 L 型外轮廓 3 个角点提取和内拐点检测
- **红色提取优化**: 改为只使用开运算（5x5 核），去掉闭运算
- **透视变换参数**: 输出尺寸改为 300×300，扩展比例 10%
- **内拐点验证**: 添加红色区域验证，确保角点在红色区域内
- **轮廓过滤调整**: 面积范围改为 200-10000

### 2026-04-26
- **红色提取链路重构**: 在调试链路中改为“灰度提轮廓 + 轮廓内红色占比筛选”
- **候选筛选分层**: 第一层仅保留“点数 > 6 + minAreaRect 长宽比”；第二层增加透视后面积占比过滤
- **透视矫正稳定性**: `warp_to_frontal()` 增加最小旋转外接矩形角点排序（TL→TR→BR→BL）后再透视
- **直线段提取增强**: 支持自适应梯度阈值（Top-K 峰）、相邻合并、闭合首尾补充合并、保留最长6条
- **接缝问题修复**: 增加闭环首尾合并与无向方向差（mod 180）比较，缓解 180/-180 分裂
- **八链码实验接入**: 新增 6 段线角度到八方向链码映射及最长4段加权模板匹配分数输出
- **主流程对齐调试链路**: `src/main.py` 已切换为与 `debug_line_extraction.py` 一致的候选筛选 + 六边提线 + 几何约束判定
- **抖动边拒绝规则**: 新增“长边高陡梯度过多”与“长边拟合残差过大”双重拒绝，过滤不规则假直线
- **多 L 严格四边形关联**: 新增“严格凸四边形 + 强相邻/强对顶 + 中心在形内 + 中心聚类共识”主筛选，`1.png` 可稳定画出四条边
- **拥挤场景保护**: 候选较多时严格筛选失败不再走宽松并查集回退，抑制 `2.png` 类误检拼框
- **逆透视射线约束**: 将 warped 线段逆透视回原图，和两臂方向/角中位线做一致性筛选后再进入关联
- **视频中心点可视化打通**: `src/test/video/test_video_line_extraction.py` 逐帧 `analyze_frame_associations()` + `QuadDrawStabilizer` + `draw_yellow_quad_and_center`（稳定后黄框与框心）
- **red.avi 实测结果**:
  - 输出 `output/red_association_center.mp4`
  - 总帧数 `6711`
  - 有最佳匹配帧数 `2959`
  - 有中心点帧数 `853`
  - 平均 best score `0.574`

### 2026-04-26（多 L 中心与观测链路）
- **问题背景**：两点 L 型时中心落在两灯条之间、四边形形状过宽、偶发内凹点当外拐点；视频 read 段（red.avi 2:00–2:20）存在 K 点抖动、侧边灯条误参与、角平分线/拟合线与视觉不一致。
- **中心与形状**：`estimate_frame_center_from_four` 分档（2/3/4+ 观测）；严格四边形 `_quad_passes_exchange_shape`（内角、边比、对边平行）；共识聚类 `_consensus_pick_best_quad`；拥挤场景下额外观测仅在落入当前四边形扩张包络内才触发整帧放弃。
- **观测构造**：`build_lbar_observation_from_six_corners`（`debug_line_extraction.py` / 视频脚本）用 6 线中相邻拟合边交点作为 K（交点离原候选过远则回退顶点）；`dir_a`/`dir_b` 固定为沿拟合边朝外；仅当角平分线与质心方向相反时翻转 `bisector`，**不再**同步翻转两臂方向（避免 c3 类“平分线指反”连带画错两臂）。
- **关联图**：`analyze_frame_associations` 用 `_ray_relation_consistency`（角平分线射线相交 + 外轮廓射线关系）替代单纯 `_adjacent_consistency`；`_bisector_ray_consistency` 对两观测各尝试 ±`bisector` 符号以兼容初始方向误差。
- **质量与回退**：`LBarObservation.match_score`；`_observation_quality` / `_pair_quality` 降权低分 L 型；`min_quad_match_score`（默认 0.30）限制进入严格四元组；fallback 仅在 3 inlier 或已有 `quad_ordered` 等路径输出中心，并经 `_center_inside_expanded_k_bounds` 过滤画面外/离谱中心。
- **可视化**：`draw_association_debug` 中 `arm_a_end`/`arm_b_end` 绘制 K 出发的两条拟合臂（BGR `(255,180,0)`）；K 到 `frame_center` 为灰色细短辅助段；四边形仍为青色闭合折线。
- **视频回归**：read 段上有中心帧数量较调参前明显减少（误报抑制优先）；完整 red.avi 指标仍以表中历史 run 为准，需单独重跑再更新数字。

### 2026-04-26（k0 射线、顺序环、视频稳定显示）
- **k0→对径角**：`corner_k2`（六角对径顶点拟合交点）；`k0_to_opposite_corner_ray`、`classify_l_pair_k0_ray_relation`（邻边 / 对角）；成对矩阵抬权底限 0.58
- **CCW 顺序环**：`ordered_quad_passes_k0_ring` 注入严格四元组枚举与 fallback 剪枝，减少误识别聚类
- **视频**：`test_video_line_extraction.py` 仅 `draw_yellow_quad_and_center`；`QuadDrawStabilizer`（连续帧、EMA、丢帧/跳变 grace）稳定后再显示框与中心
- **包入口**：`from calculation import analyze_frame_associations, QuadDrawStabilizer, …` 见 `geometric.md`

### 2026-04-15
- **重构文档体系**: 采用递归式访问结构，创建 MAP.md，精简 docs/
- **保留双 agent**: code-writer（算法开发）+ code-reviewer（文档维护）
- **强化递归式访问**: 先读 MAP.md → 定向读取模块 → 增量更新
- **更新场景信息**: 明确 1 个静止兑换站，有红色干扰，视频部分帧无目标

### 历史记录
- **2026-04-15**: 创建 Agent 管理体系（CLAUDE.md、agents/、rules/）
- **2026-03-21**: 实现链码算法（test_chaincode_perspective.py）
- **初始**: 项目启动，定义几何模型和协作规范

## 已完成

- ✅ 基于链码的 L 型灯条识别算法
- ✅ `main.py` 与 `debug_line_extraction.py` 主链路对齐（候选筛选 + 六边提线 + 八链码分数 + 几何约束）
- ✅ 透视变换与矫正（300×300 输出）
- ✅ L 型外轮廓角点提取（3 个角点）
- ✅ 内拐点检测（Shi-Tomasi + 红色区域验证）
- ✅ 红色提取优化（只使用开运算，5x5 核）
- ✅ Docker 环境配置
- ✅ Agent 管理体系
- ✅ 递归式文档结构
