#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "realsense_subscriber/realsense_subscriber_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<realsense_subscriber::RealsenseSubscriberNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
