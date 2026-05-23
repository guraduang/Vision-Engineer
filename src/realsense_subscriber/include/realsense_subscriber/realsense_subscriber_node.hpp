#ifndef REALSENSE_SUBSCRIBER__REALSENSE_SUBSCRIBER_NODE_HPP_
#define REALSENSE_SUBSCRIBER__REALSENSE_SUBSCRIBER_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "realsense_subscriber/parameters.hpp"
#include "realsense_subscriber/vision_pipeline.hpp"

namespace realsense_subscriber
{

/**
 * @brief ROS2节点模块：负责参数加载、消息订阅与位姿发布。
 */
class RealsenseSubscriberNode : public rclcpp::Node
{
public:
  RealsenseSubscriberNode();

private:
  /**
   * @brief 图像回调，执行视觉主链路并发布位姿。
   * @param[in] msg ROS图像消息。
   */
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

  /**
   * @brief 构建相机模型参数。
   * @param[in] camera_matrix_values 3x3内参展开数组。
   * @param[in] distortion_coefficients 畸变参数数组。
   * @return 相机模型结构。
   */
  static CameraModel build_camera_model(
    const std::vector<double> & camera_matrix_values,
    const std::vector<double> & distortion_coefficients);

  /**
   * @brief 发布位姿消息。
   * @param[in] header 输入图像消息头。
   * @param[in] pose_solution 位姿求解结果。
   */
  void publish_pose(
    const std_msgs::msg::Header & header,
    const PoseSolution & pose_solution);

  /**
   * @brief 发布调试图像（原图叠加角点 + 二值图 + 边缘图）。
   * @param[in] header 输入图像消息头。
   * @param[in] bgr_image 原始BGR图像。
   * @param[in] debug_frame 功能模块输出的中间结果。
   * @param[in] pose_success 位姿是否求解成功。
   */
  void publish_debug_image(
    const std_msgs::msg::Header & header,
    const cv::Mat & bgr_image,
    const DebugFrame & debug_frame,
    bool pose_success);

  std::string image_topic_;
  std::string pose_topic_;
  std::string debug_image_topic_;
  bool enable_debug_visualization_{false};
  std::unique_ptr<VisionPipeline> vision_pipeline_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_publisher_;
};

}  // namespace realsense_subscriber

#endif  // REALSENSE_SUBSCRIBER__REALSENSE_SUBSCRIBER_NODE_HPP_
