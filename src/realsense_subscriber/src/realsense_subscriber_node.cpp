#include <algorithm>
#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include "realsense_subscriber/realsense_subscriber_node.hpp"

namespace realsense_subscriber
{

RealsenseSubscriberNode::RealsenseSubscriberNode()
: Node("realsense_subscriber_node")
{
  // 1) 统一从参数模块加载当前节点实际使用的参数。
  const SubscriberParameters params = load_subscriber_parameters(*this);
  image_topic_ = params.image_topic;
  pose_topic_ = params.pose_topic;
  debug_image_topic_ = params.debug_image_topic;
  enable_debug_visualization_ = params.enable_debug_visualization;

  CameraModel camera_model = build_camera_model(
    params.camera_matrix_values,
    params.distortion_coefficients);
  vision_pipeline_ = std::make_unique<VisionPipeline>(params.processing_config, camera_model);

  // 2) 初始化ROS通信实体。
  pose_publisher_ = create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic_, rclcpp::QoS(10));
  debug_image_publisher_ =
    create_publisher<sensor_msgs::msg::Image>(debug_image_topic_, rclcpp::QoS(10));
  image_subscription_ = create_subscription<sensor_msgs::msg::Image>(
    image_topic_, rclcpp::SensorDataQoS(),
    std::bind(&RealsenseSubscriberNode::image_callback, this, std::placeholders::_1));

  RCLCPP_INFO(
    get_logger(),
    "realsense_subscriber started: image_topic=%s pose_topic=%s debug_topic=%s debug_enabled=%s",
    image_topic_.c_str(),
    pose_topic_.c_str(),
    debug_image_topic_.c_str(),
    enable_debug_visualization_ ? "true" : "false");
  if (!enable_debug_visualization_) {
    RCLCPP_WARN(
      get_logger(),
      "Debug visualization is disabled. Set enable_debug_visualization=true to publish debug image.");
  }
}

void RealsenseSubscriberNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // 3) 将ROS图像转成OpenCV格式并交给功能模块求解位姿。
  cv::Mat bgr_image;
  try {
    bgr_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
  } catch (const cv_bridge::Exception & exception) {
    RCLCPP_WARN(get_logger(), "cv_bridge conversion failed: %s", exception.what());
    return;
  }

  PoseSolution pose_solution{};
  DebugFrame debug_frame{};
  const bool pose_success = vision_pipeline_->estimate_pose(
    bgr_image, pose_solution, enable_debug_visualization_ ? &debug_frame : nullptr);

  if (enable_debug_visualization_) {
    publish_debug_image(msg->header, bgr_image, debug_frame, pose_success);
    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Debug image published to %s (candidate=%s pnp=%s reproj=%.3f, subscribers=%zu)",
      debug_image_topic_.c_str(),
      debug_frame.has_candidate ? "true" : "false",
      debug_frame.pnp_solved ? "true" : "false",
      debug_frame.reprojection_error,
      debug_image_publisher_->get_subscription_count());
    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Stage=%s contours(total/area/hierarchy/chain/corner)=%d/%d/%d/%d/%d",
      debug_frame.failure_stage.c_str(),
      debug_frame.contours_total,
      debug_frame.contours_area_pass,
      debug_frame.contours_hierarchy_pass,
      debug_frame.contours_chain_code_pass,
      debug_frame.contours_corner_pass);
  }
  if (!pose_success) {
    return;
  }
  publish_pose(msg->header, pose_solution);
}

CameraModel RealsenseSubscriberNode::build_camera_model(
  const std::vector<double> & camera_matrix_values,
  const std::vector<double> & distortion_coefficients)
{
  CameraModel camera_model;
  camera_model.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
  if (camera_matrix_values.size() == 9U) {
    std::memcpy(camera_model.camera_matrix.data, camera_matrix_values.data(), 9 * sizeof(double));
  }

  camera_model.distortion_coefficients = cv::Mat::zeros(
    static_cast<int>(distortion_coefficients.size()), 1, CV_64F);
  if (!distortion_coefficients.empty()) {
    std::memcpy(
      camera_model.distortion_coefficients.data, distortion_coefficients.data(),
      distortion_coefficients.size() * sizeof(double));
  }
  return camera_model;
}

void RealsenseSubscriberNode::publish_pose(
  const std_msgs::msg::Header & header,
  const PoseSolution & pose_solution)
{
  // 4) 将旋转向量转换为四元数并封装PoseStamped发布。
  cv::Mat rotation_matrix;
  cv::Rodrigues(pose_solution.rotation_vector, rotation_matrix);

  const double trace = rotation_matrix.at<double>(0, 0) +
    rotation_matrix.at<double>(1, 1) + rotation_matrix.at<double>(2, 2);
  const double qw = std::sqrt(std::max(0.0, 1.0 + trace)) * 0.5;
  const double qx = (rotation_matrix.at<double>(2, 1) - rotation_matrix.at<double>(1, 2)) /
    (4.0 * std::max(qw, 1e-9));
  const double qy = (rotation_matrix.at<double>(0, 2) - rotation_matrix.at<double>(2, 0)) /
    (4.0 * std::max(qw, 1e-9));
  const double qz = (rotation_matrix.at<double>(1, 0) - rotation_matrix.at<double>(0, 1)) /
    (4.0 * std::max(qw, 1e-9));

  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header = header;
  pose_msg.pose.position.x = pose_solution.translation_vector[0];
  pose_msg.pose.position.y = pose_solution.translation_vector[1];
  pose_msg.pose.position.z = pose_solution.translation_vector[2];
  pose_msg.pose.orientation.w = qw;
  pose_msg.pose.orientation.x = qx;
  pose_msg.pose.orientation.y = qy;
  pose_msg.pose.orientation.z = qz;
  pose_publisher_->publish(pose_msg);
}

void RealsenseSubscriberNode::publish_debug_image(
  const std_msgs::msg::Header & header,
  const cv::Mat & bgr_image,
  const DebugFrame & debug_frame,
  bool pose_success)
{
  cv::Mat overlay = bgr_image.clone();
  for (size_t index = 0; index < debug_frame.selected_corners.size(); ++index) {
    const cv::Point2f & corner = debug_frame.selected_corners[index];
    cv::circle(overlay, corner, 5, cv::Scalar(0, 255, 0), 2);
    cv::putText(
      overlay,
      std::to_string(index),
      corner + cv::Point2f(6.0F, -6.0F),
      cv::FONT_HERSHEY_SIMPLEX,
      0.6,
      cv::Scalar(0, 255, 0),
      2);
  }

  const std::string status_text = pose_success ? "POSE: OK" : "POSE: FAIL";
  cv::putText(
    overlay,
    status_text,
    cv::Point(12, 28),
    cv::FONT_HERSHEY_SIMPLEX,
    0.8,
    pose_success ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 255),
    2);
  cv::putText(
    overlay,
    "REPROJ: " + std::to_string(debug_frame.reprojection_error),
    cv::Point(12, 58),
    cv::FONT_HERSHEY_SIMPLEX,
    0.7,
    cv::Scalar(255, 255, 0),
    2);
  cv::putText(
    overlay,
    "STAGE: " + debug_frame.failure_stage,
    cv::Point(12, 88),
    cv::FONT_HERSHEY_SIMPLEX,
    0.7,
    cv::Scalar(0, 255, 255),
    2);
  cv::putText(
    overlay,
    "C(total/a/h/c/c): " + std::to_string(debug_frame.contours_total) + "/" +
      std::to_string(debug_frame.contours_area_pass) + "/" +
      std::to_string(debug_frame.contours_hierarchy_pass) + "/" +
      std::to_string(debug_frame.contours_chain_code_pass) + "/" +
      std::to_string(debug_frame.contours_corner_pass),
    cv::Point(12, 118),
    cv::FONT_HERSHEY_SIMPLEX,
    0.55,
    cv::Scalar(255, 255, 255),
    2);

  cv::Mat binary_bgr;
  cv::Mat edge_bgr;
  if (!debug_frame.red_binary_mask.empty()) {
    cv::cvtColor(debug_frame.red_binary_mask, binary_bgr, cv::COLOR_GRAY2BGR);
  } else {
    binary_bgr = cv::Mat::zeros(bgr_image.size(), CV_8UC3);
  }
  if (!debug_frame.edge_image.empty()) {
    cv::cvtColor(debug_frame.edge_image, edge_bgr, cv::COLOR_GRAY2BGR);
  } else {
    edge_bgr = cv::Mat::zeros(bgr_image.size(), CV_8UC3);
  }

  cv::putText(
    binary_bgr, "Binary Mask", cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.8,
    cv::Scalar(255, 255, 255), 2);
  cv::putText(
    edge_bgr, "Canny Edge", cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.8,
    cv::Scalar(255, 255, 255), 2);
  cv::putText(
    overlay, "Original + Corners", cv::Point(12, 88), cv::FONT_HERSHEY_SIMPLEX, 0.8,
    cv::Scalar(255, 255, 255), 2);

  cv::Mat top_row;
  cv::hconcat(overlay, binary_bgr, top_row);
  cv::Mat debug_canvas;
  cv::hconcat(top_row, edge_bgr, debug_canvas);

  auto debug_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, debug_canvas)
                     .toImageMsg();
  debug_image_publisher_->publish(*debug_msg);
}

}  // namespace realsense_subscriber
