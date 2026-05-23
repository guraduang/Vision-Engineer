#include "realsense_subscriber/parameters.hpp"

namespace realsense_subscriber
{

SubscriberParameters load_subscriber_parameters(rclcpp::Node & node)
{
  SubscriberParameters parameters;

  parameters.image_topic = node.declare_parameter<std::string>(
    "image_topic", parameters.image_topic);
  parameters.pose_topic = node.declare_parameter<std::string>(
    "pose_topic", parameters.pose_topic);
  parameters.debug_image_topic = node.declare_parameter<std::string>(
    "debug_image_topic", parameters.debug_image_topic);
  parameters.enable_debug_visualization = node.declare_parameter<bool>(
    "enable_debug_visualization", parameters.enable_debug_visualization);

  parameters.processing_config.marker_size_meter = node.declare_parameter<double>(
    "marker_size_meter", parameters.processing_config.marker_size_meter);
  parameters.processing_config.red_minus_blue_threshold = node.declare_parameter<int>(
    "red_minus_blue_threshold", parameters.processing_config.red_minus_blue_threshold);
  parameters.processing_config.binary_threshold = node.declare_parameter<int>(
    "binary_threshold", parameters.processing_config.binary_threshold);
  parameters.processing_config.canny_low_threshold = node.declare_parameter<int>(
    "canny_low_threshold", parameters.processing_config.canny_low_threshold);
  parameters.processing_config.canny_high_threshold = node.declare_parameter<int>(
    "canny_high_threshold", parameters.processing_config.canny_high_threshold);
  parameters.processing_config.min_contour_area = node.declare_parameter<double>(
    "min_contour_area", parameters.processing_config.min_contour_area);
  parameters.processing_config.max_reprojection_error = node.declare_parameter<double>(
    "max_reprojection_error", parameters.processing_config.max_reprojection_error);
  parameters.processing_config.min_chain_code_transitions = node.declare_parameter<int>(
    "min_chain_code_transitions", parameters.processing_config.min_chain_code_transitions);
  parameters.processing_config.max_chain_code_transitions = node.declare_parameter<int>(
    "max_chain_code_transitions", parameters.processing_config.max_chain_code_transitions);

  parameters.camera_matrix_values = node.declare_parameter<std::vector<double>>(
    "camera_matrix", parameters.camera_matrix_values);
  parameters.distortion_coefficients = node.declare_parameter<std::vector<double>>(
    "distortion_coefficients", parameters.distortion_coefficients);

  return parameters;
}

}  // namespace realsense_subscriber
