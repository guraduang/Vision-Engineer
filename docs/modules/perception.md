# 感知模块

> 从图像中提取红色区域并检测轮廓

## 输入/输出协议

### extract_red_mask(image: np.ndarray, threshold: int = 30) -> np.ndarray
- **输入**: BGR 图像
- **输出**: 二值掩码（0/255）
- **方法**: R-B 通道差值法
- **位置**: `src/perception/red_extractor.py`

```python
def extract_red_mask(image: np.ndarray, threshold: int = 30) -> np.ndarray:
    """提取红色区域（R-B 差值法）"""
    b, g, r = cv2.split(image)
    red_diff = cv2.subtract(r, b)
    _, mask = cv2.threshold(red_diff, threshold, 255, cv2.THRESH_BINARY)
    return mask
```

### extract_red_mask_from_gray_contours(
image: np.ndarray,
gray_threshold: int = 20,
red_diff_threshold: int = 30,
red_ratio_threshold: float = 0.5,
min_contour_area: float = 20.0
) -> np.ndarray
- **输入**: BGR 图像 + 灰度阈值 + 红色判定阈值
- **输出**: 二值掩码（0/255）
- **方法**: 先灰度阈值提轮廓，再按轮廓内红色像素占比筛选
- **位置**: `src/perception/red_extractor.py`

```python
def extract_red_mask_from_gray_contours(image: np.ndarray,
                                        gray_threshold: int = 20,
                                        red_diff_threshold: int = 30,
                                        red_ratio_threshold: float = 0.5,
                                        min_contour_area: float = 20.0) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, gray_binary = cv2.threshold(gray, gray_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(gray_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # ... 逐轮廓统计 R-B > red_diff_threshold 的像素占比并筛选
    return mask
```

### find_contours(mask: np.ndarray) -> List[np.ndarray]
- **输入**: 二值掩码
- **输出**: 轮廓列表
- **位置**: `src/perception/contour_detector.py`

### filter_contours_by_area(contours, min_area=100, max_area=50000) -> List[np.ndarray]
- **输入**: 轮廓列表，面积范围
- **输出**: 筛选后的轮廓列表
- **位置**: `src/perception/contour_detector.py`

### compute_warped_area_ratio(
image: np.ndarray,
contour: np.ndarray,
warp_fn,
mask_fn,
warped_min_area: float = 50.0
) -> Optional[float]
- **输入**: 原图、原图候选轮廓、透视函数、掩码函数
- **输出**: 透视后最大候选轮廓面积占比（面积 / warped 图总面积）
- **方法**: 透视矫正 -> 提掩码 -> 找轮廓 -> 取最大轮廓面积占比
- **位置**: `src/perception/contour_detector.py`

### filter_contours_by_warped_area_ratio(
image: np.ndarray,
contours: List[np.ndarray],
warp_fn,
mask_fn,
min_area_ratio: float = 0.10,
max_area_ratio: float = 0.85,
warped_min_area: float = 50.0
) -> List[np.ndarray]
- **输入**: 第一层候选轮廓 + 透视函数 + 掩码函数 + 面积占比阈值
- **输出**: 通过第二层筛选的候选轮廓
- **方法**: 对每个候选计算透视后面积占比，仅保留占比在阈值区间的目标
- **位置**: `src/perception/contour_detector.py`
- **说明**: 当前主流程 `src/main.py` 使用 `debug_line_extraction.py` 中的
  `filter_candidates_for_line_extraction()`（含自适应面积上限 + warped 占比 + recover 路径）。

## 关键参数

| 参数 | 默认值 | 说明 | 行号 |
|------|--------|------|------|
| threshold | 30 | R-B 差值阈值 | L13 |
| gray_threshold | 20 | 灰度二值化阈值（轮廓提取） | 代码中 |
| red_diff_threshold | 30 | 红色判定阈值（R-B） | 代码中 |
| red_ratio_threshold | 0.5 | 轮廓内红色像素占比阈值 | 代码中 |
| min_contour_area | 20 | 最小轮廓面积 | 代码中 |
| min_area | 100 | 最小轮廓面积（第一层） | 代码中 |
| max_area | 50000 | 最大轮廓面积（第一层） | 代码中 |
| min_points | 7 | 几何筛选最小轮廓点数（保留点数 > 6） | 代码中 |
| max_side_ratio | 6.0 | 几何筛选最大长宽比（minAreaRect） | 代码中 |
| min_area_ratio | 0.10 | 透视后面积占比下限（第二层） | 代码中 |
| max_area_ratio | 0.85 | 透视后面积占比上限（第二层） | 代码中 |
| warped_min_area | 50 | 透视图最小候选面积 | 代码中 |
| line_extract_max_area_ratio | 0.585 | 主流程二层筛选 warped 占比上限 | `debug_line_extraction.py` |
| line_extract_recover_min_area | 180 | warp 为空时恢复到原图轮廓的最小面积 | `debug_line_extraction.py` |

## 算法流程

1. **通道分离**: 分离 BGR 三通道
2. **差值计算**: R - B，突出红色区域
3. **二值化**: 阈值 30
4. **轮廓检测**: OpenCV findContours
5. **第一层筛选**: 面积范围 + 几何筛选（仅保留点数 > 6 且 minAreaRect 长宽比达标）
6. **第二层筛选（主流程）**: `filter_candidates_for_line_extraction()`（自适应面积上限 + warped 占比 + recover）

## 可替换方案

### HSV 色彩空间法
```python
def extract_red_mask_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 红色在 HSV 中跨越 0 度
    mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    return cv2.bitwise_or(mask1, mask2)
```

## 当前问题

- 侧视角（2.png）有 5 个灯条（4 正面 + 1 侧面），需准确识别
- 侧视图（3.png）仅 1 个侧边灯条，形状变形严重
- 场景中存在其他红色灯条干扰
- 视频中部分帧无兑换站，需避免误检

## 优化方向

- 自适应阈值（根据图像亮度调整）
- 形状约束（长宽比、角度等）过滤干扰
- 时序一致性（视频中兑换站静止，利用时序信息）
- 多尺度检测（处理不同视角的灯条）
