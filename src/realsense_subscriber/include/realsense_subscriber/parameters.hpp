#ifndef REALSENSE_SUBSCRIBER__PARAMETERS_HPP_
#define REALSENSE_SUBSCRIBER__PARAMETERS_HPP_

#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "realsense_subscriber/vision_pipeline.hpp"

namespace realsense_subscriber
{

/**
 * @brief 节点参数集合，仅包含当前 realsense_subscriber 实际使用的变量。
 */
struct SubscriberParameters
{
  std::string image_topic{"/camera/camera/color/image_raw"};
  std::string pose_topic{"/target_pose"};
  std::string debug_image_topic{"/realsense_subscriber/debug_image"};
  bool enable_debug_visualization{true};

  ProcessingConfig processing_config{};
  std::vector<double> camera_matrix_values{
    615.0, 0.0, 320.0,
    0.0, 615.0, 240.0,
    0.0, 0.0, 1.0};
  std::vector<double> distortion_coefficients{0.0, 0.0, 0.0, 0.0, 0.0};
};

/**
 * @brief 从 ROS2 参数服务器读取节点参数。
 * @param[in,out] node 当前节点。
 * @return 读取后的参数集合。
 */
SubscriberParameters load_subscriber_parameters(rclcpp::Node & node);

}  // namespace realsense_subscriber

#endif  // REALSENSE_SUBSCRIBER__PARAMETERS_HPP_
