# 接口模块

> ROS2 节点，订阅相机话题，发布检测结果

## 状态

🚧 **待实现**（当前优先 Python 算法验证）

## 规划接口

### 订阅话题
- `/camera/color/image_raw` (sensor_msgs/Image) - RGB 图像
- `/camera/depth/image_raw` (sensor_msgs/Image) - 深度图像
- `/camera/camera_info` (sensor_msgs/CameraInfo) - 相机内参

### 发布话题
- `/vision/l_shapes` (自定义消息) - 检测到的 L 型灯条
- `/vision/pose` (geometry_msgs/PoseStamped) - 位姿估计结果
- `/vision/debug_image` (sensor_msgs/Image) - 可视化调试图像

## 消息定义

### LShapeArray.msg
```
Header header
LShape[] shapes
```

### LShape.msg
```
geometry_msgs/Point2D[] corners  # 6 个角点
float32 confidence               # 置信度
int32 id                         # 灯条 ID
```

## 参考实现

**C++ ROS2 节点**: `src/realsense_subscriber/`

| 文件 | 功能 |
|------|------|
| `realsense_subscriber_node.hpp/cpp` | ROS2 节点主体 |
| `vision_pipeline.hpp/cpp` | 视觉处理管道 |
| `parameters.hpp/cpp` | 参数管理 |

## 集成方案

### Python 节点（推荐）
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.subscription = self.create_subscription(
            Image, '/camera/color/image_raw', 
            self.image_callback, 10)
        self.bridge = CvBridge()
    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # 调用 detect_l_shapes()
        pass
```

### C++ 节点（高性能）
- 复用现有 `realsense_subscriber` 框架
- 集成 Python 算法（pybind11）
- 或用 C++ 重写核心算法

## 待办事项

- [ ] 定义自定义消息类型
- [ ] 实现 Python ROS2 节点
- [ ] 集成链码检测算法
- [ ] 添加 PnP 位姿估计
- [ ] 性能优化（实时性）
