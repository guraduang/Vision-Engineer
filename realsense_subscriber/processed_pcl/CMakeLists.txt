cmake_minimum_required(VERSION 3.8)
project(processed_pcl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_BUILD_TYPE DEBUG)

include_directories(/opt/ros/humble/include)# 查找依赖项  
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
#find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(message_filters REQUIRED)
#find_package(cv_bridge REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(realsense_msgs REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${message_filters_INCLUDE_DIRS})
include_directories(${pcl_msgs_INCLUDE_DIRS})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(processed src/processed.cpp)

ament_target_dependencies(processed
  rclcpp
  std_msgs
  sensor_msgs
  #OpenCV
  PCL
  #cv_bridge
  geometry_msgs
  visualization_msgs
  tf2_geometry_msgs
  realsense_msgs
)

target_link_libraries(processed
  ${PCL_LIBRARIES}
)

target_include_directories(processed PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${message_filters_INCLUDE_DIRS}
  ${pcl_msgs_INCLUDE_DIRS}
)

install(TARGETS
  processed
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
