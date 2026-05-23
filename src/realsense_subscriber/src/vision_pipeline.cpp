#include "realsense_subscriber/vision_pipeline.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace realsense_subscriber
{

// 构造时注入处理参数与相机模型，后续处理过程只读使用。
VisionPipeline::VisionPipeline(
  const ProcessingConfig & processing_config,
  const CameraModel & camera_model)
: processing_config_(processing_config), camera_model_(camera_model)
{
}

bool VisionPipeline::estimate_pose(
  const cv::Mat & bgr_image,
  PoseSolution & pose_solution,
  DebugFrame * debug_frame) const
{
  if (bgr_image.empty()) {
    if (debug_frame != nullptr) {
      debug_frame->failure_stage = "EMPTY_IMAGE";
    }
    return false;
  }

  // 主链路：红色掩膜 -> 边缘轮廓 -> 角点 -> PnP -> 误差门限。
  const cv::Mat red_binary_mask = extract_red_binary_mask(bgr_image);
  cv::Mat edge_image;
  cv::Canny(
    red_binary_mask, edge_image, processing_config_.canny_low_threshold,
    processing_config_.canny_high_threshold);
  if (debug_frame != nullptr) {
    debug_frame->red_binary_mask = red_binary_mask.clone();
    debug_frame->edge_image = edge_image.clone();
    debug_frame->selected_corners.clear();
    debug_frame->has_candidate = false;
    debug_frame->pnp_solved = false;
    debug_frame->reprojection_error = 0.0;
    debug_frame->failure_stage = "CONTOUR_SELECTION";
    debug_frame->contours_total = 0;
    debug_frame->contours_area_pass = 0;
    debug_frame->contours_hierarchy_pass = 0;
    debug_frame->contours_chain_code_pass = 0;
    debug_frame->contours_corner_pass = 0;
  }

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(edge_image, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
  if (debug_frame != nullptr) {
    debug_frame->contours_total = static_cast<int>(contours.size());
  }

  std::vector<cv::Point2f> image_corners;
  if (!select_target_corners(contours, hierarchy, image_corners, debug_frame)) {
    return false;
  }
  if (debug_frame != nullptr) {
    debug_frame->selected_corners = image_corners;
    debug_frame->has_candidate = true;
  }

  if (!solve_target_pose_pnp(image_corners, pose_solution)) {
    if (debug_frame != nullptr) {
      debug_frame->failure_stage = "PNP_SOLVE";
    }
    return false;
  }
  if (debug_frame != nullptr) {
    debug_frame->pnp_solved = true;
    debug_frame->reprojection_error = pose_solution.reprojection_error;
  }

  if (pose_solution.reprojection_error > processing_config_.max_reprojection_error) {
    if (debug_frame != nullptr) {
      debug_frame->failure_stage = "REPROJECTION_GATE";
    }
    return false;
  }
  if (debug_frame != nullptr) {
    debug_frame->failure_stage = "SUCCESS";
  }
  return true;
}

cv::Mat VisionPipeline::extract_red_binary_mask(const cv::Mat & bgr_image) const
{
  // R-B 对红色目标更敏感，再叠加双阈值与形态学去噪。
  std::vector<cv::Mat> channels;
  cv::split(bgr_image, channels);
  cv::Mat red_minus_blue;
  cv::subtract(channels[2], channels[0], red_minus_blue);

  cv::Mat red_mask;
  cv::threshold(
    red_minus_blue, red_mask, processing_config_.red_minus_blue_threshold, 255.0,
    cv::THRESH_BINARY);
  cv::threshold(
    red_mask, red_mask, processing_config_.binary_threshold, 255.0, cv::THRESH_BINARY);

  cv::morphologyEx(
    red_mask, red_mask, cv::MORPH_OPEN,
    cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
  cv::morphologyEx(
    red_mask, red_mask, cv::MORPH_CLOSE,
    cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

  return red_mask;
}

bool VisionPipeline::select_target_corners(
  const std::vector<std::vector<cv::Point>> & contours,
  const std::vector<cv::Vec4i> & hierarchy,
  std::vector<cv::Point2f> & image_corners,
  DebugFrame * debug_frame) const
{
  // 在候选轮廓中选择评分最优的一个，避免多目标时随机命中。
  bool found = false;
  double best_score = std::numeric_limits<double>::max();

  for (size_t contour_index = 0; contour_index < contours.size(); ++contour_index) {
    const auto & contour = contours[contour_index];
    if (contour.empty()) {
      continue;
    }
    if (cv::contourArea(contour) < processing_config_.min_contour_area) {
      continue;
    }
    if (debug_frame != nullptr) {
      ++debug_frame->contours_area_pass;
    }
    // 保留无内包轮廓，过滤有子轮廓的复杂区域。
    if (hierarchy.size() == contours.size() && hierarchy[contour_index][2] != -1) {
      continue;
    }
    if (debug_frame != nullptr) {
      ++debug_frame->contours_hierarchy_pass;
    }
    if (!is_chain_code_candidate(contour)) {
      continue;
    }
    if (debug_frame != nullptr) {
      ++debug_frame->contours_chain_code_pass;
    }

    std::vector<cv::Point2f> corners;
    if (!extract_ordered_corners(contour, corners)) {
      continue;
    }
    if (debug_frame != nullptr) {
      ++debug_frame->contours_corner_pass;
    }

    const double score = contour_score(contour, corners);
    if (score < best_score) {
      best_score = score;
      image_corners = corners;
      found = true;
    }
  }

  return found;
}

bool VisionPipeline::extract_ordered_corners(
  const std::vector<cv::Point> & contour,
  std::vector<cv::Point2f> & ordered_corners) const
{
  // 先多边形逼近，点数多于4时退化为最小外接矩形，保证可进入PnP。
  std::vector<cv::Point> approx_polygon;
  cv::approxPolyDP(contour, approx_polygon, 0.02 * cv::arcLength(contour, true), true);
  if (approx_polygon.size() < 4U) {
    return false;
  }

  std::vector<cv::Point2f> corners;
  if (approx_polygon.size() == 4U) {
    corners.reserve(4);
    for (const auto & point : approx_polygon) {
      corners.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
    }
  } else {
    cv::RotatedRect min_rect = cv::minAreaRect(approx_polygon);
    std::array<cv::Point2f, 4> rect_points{};
    min_rect.points(rect_points.data());
    corners.assign(rect_points.begin(), rect_points.end());
  }

  order_corners_clockwise(corners);
  if (corners.size() != 4U) {
    return false;
  }
  ordered_corners = corners;
  return true;
}

bool VisionPipeline::is_chain_code_candidate(const std::vector<cv::Point> & contour) const
{
  if (contour.size() < 8U) {
    return false;
  }

  // 将轮廓边界方向离散到8方向链码，统计方向跳变次数作为形状约束。
  std::vector<int> chain_codes;
  chain_codes.reserve(contour.size());
  for (size_t index = 1; index < contour.size(); ++index) {
    const int dx = contour[index].x - contour[index - 1].x;
    const int dy = contour[index].y - contour[index - 1].y;
    if (dx == 0 && dy == 0) {
      continue;
    }
    const double angle_deg = normalize_angle(
      std::atan2(static_cast<double>(dy), static_cast<double>(dx)) * 180.0 / CV_PI);
    const int chain_code = static_cast<int>(std::lround(angle_deg / 45.0)) % 8;
    chain_codes.push_back(chain_code);
  }

  if (chain_codes.empty()) {
    return false;
  }

  int transition_count = 0;
  for (size_t index = 1; index < chain_codes.size(); ++index) {
    if (chain_codes[index] != chain_codes[index - 1]) {
      ++transition_count;
    }
  }

  return transition_count >= processing_config_.min_chain_code_transitions &&
         transition_count <= processing_config_.max_chain_code_transitions;
}

bool VisionPipeline::solve_target_pose_pnp(
  const std::vector<cv::Point2f> & image_corners,
  PoseSolution & pose_solution) const
{
  if (image_corners.size() != 4U) {
    return false;
  }

  // 目标模型采用边长 marker_size_meter 的平面方形（Z=0）。
  const float half_size = static_cast<float>(processing_config_.marker_size_meter / 2.0);
  const std::vector<cv::Point3f> object_points{
    {-half_size, -half_size, 0.0F},
    {half_size, -half_size, 0.0F},
    {half_size, half_size, 0.0F},
    {-half_size, half_size, 0.0F}};

  const bool solved = cv::solvePnP(
    object_points,
    image_corners,
    camera_model_.camera_matrix,
    camera_model_.distortion_coefficients,
    pose_solution.rotation_vector,
    pose_solution.translation_vector,
    false,
    cv::SOLVEPNP_IPPE);
  if (!solved) {
    return false;
  }

  // 用重投影误差评估解算质量，供上游做门限裁剪。
  std::vector<cv::Point2f> reprojected_points;
  cv::projectPoints(
    object_points,
    pose_solution.rotation_vector,
    pose_solution.translation_vector,
    camera_model_.camera_matrix,
    camera_model_.distortion_coefficients,
    reprojected_points);

  pose_solution.reprojection_error = 0.0;
  for (size_t index = 0; index < image_corners.size(); ++index) {
    pose_solution.reprojection_error += cv::norm(image_corners[index] - reprojected_points[index]);
  }
  pose_solution.reprojection_error /= static_cast<double>(image_corners.size());
  return true;
}

void VisionPipeline::order_corners_clockwise(std::vector<cv::Point2f> & corners)
{
  // 先按相对中心极角排序，再旋转到左上角起点，保证角点顺序稳定。
  cv::Point2f center(0.0F, 0.0F);
  for (const auto & point : corners) {
    center.x += point.x;
    center.y += point.y;
  }
  center.x /= static_cast<float>(corners.size());
  center.y /= static_cast<float>(corners.size());

  std::sort(
    corners.begin(), corners.end(),
    [&center](const cv::Point2f & left, const cv::Point2f & right) {
      return std::atan2(left.y - center.y, left.x - center.x) <
             std::atan2(right.y - center.y, right.x - center.x);
    });

  const auto top_left_iterator = std::min_element(
    corners.begin(), corners.end(),
    [](const cv::Point2f & left, const cv::Point2f & right) {
      return (left.x + left.y) < (right.x + right.y);
    });
  std::rotate(corners.begin(), top_left_iterator, corners.end());
}

double VisionPipeline::contour_score(
  const std::vector<cv::Point> & contour,
  const std::vector<cv::Point2f> & corners)
{
  // 紧致度越低越接近规则目标，同时惩罚非四角情况。
  const double area = std::max(std::abs(cv::contourArea(contour)), 1.0);
  const double perimeter = cv::arcLength(contour, true);
  const double compactness = (perimeter * perimeter) / area;
  const double corner_penalty = std::abs(static_cast<int>(corners.size()) - 4) * 100.0;
  return compactness + corner_penalty;
}

double VisionPipeline::normalize_angle(double angle_deg)
{
  // 将角度归一化到 [0, 360) 便于离散链码。
  while (angle_deg < 0.0) {
    angle_deg += 360.0;
  }
  while (angle_deg >= 360.0) {
    angle_deg -= 360.0;
  }
  return angle_deg;
}

}  // namespace realsense_subscriber
