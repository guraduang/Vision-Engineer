#ifndef REALSENSE_SUBSCRIBER__VISION_PIPELINE_HPP_
#define REALSENSE_SUBSCRIBER__VISION_PIPELINE_HPP_

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace realsense_subscriber
{

/**
 * @brief 图像与几何处理参数集合。
 */
struct ProcessingConfig
{
  double marker_size_meter{0.06};
  int red_minus_blue_threshold{60};
  int binary_threshold{120};
  int canny_low_threshold{50};
  int canny_high_threshold{150};
  double min_contour_area{180.0};
  double max_reprojection_error{5.0};
  int min_chain_code_transitions{4};
  int max_chain_code_transitions{10};
};

/**
 * @brief 相机模型参数。
 */
struct CameraModel
{
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
};

/**
 * @brief 位姿求解结果。
 */
struct PoseSolution
{
  cv::Vec3d rotation_vector;
  cv::Vec3d translation_vector;
  double reprojection_error{0.0};
};

/**
 * @brief 调试可视化所需的中间结果。
 */
struct DebugFrame
{
  cv::Mat red_binary_mask;
  cv::Mat edge_image;
  std::vector<cv::Point2f> selected_corners;
  bool has_candidate{false};
  bool pnp_solved{false};
  double reprojection_error{0.0};
  std::string failure_stage{"NONE"};
  int contours_total{0};
  int contours_area_pass{0};
  int contours_hierarchy_pass{0};
  int contours_chain_code_pass{0};
  int contours_corner_pass{0};
};

/**
 * @brief 视觉处理模块：完成红色目标提取、角点筛选与PnP位姿解算。
 */
class VisionPipeline
{
public:
  VisionPipeline(const ProcessingConfig & processing_config, const CameraModel & camera_model);

  /**
   * @brief 在BGR图像中检测目标并求解位姿。
   * @param[in] bgr_image 输入BGR图像。
   * @param[out] pose_solution 求解成功时返回位姿与重投影误差。
   * @param[out] debug_frame 调试信息（可空）。
   * @return 成功返回true；失败返回false。
   * @note 失败条件：无有效轮廓、角点不足、PnP失败或重投影误差超阈值。
   */
  bool estimate_pose(
    const cv::Mat & bgr_image,
    PoseSolution & pose_solution,
    DebugFrame * debug_frame = nullptr) const;

private:
  /**
   * @brief 提取红色目标二值掩膜。
   * @param[in] bgr_image 输入BGR图像。
   * @return 红色二值图。
   */
  cv::Mat extract_red_binary_mask(const cv::Mat & bgr_image) const;

  /**
   * @brief 从轮廓中筛选最优候选角点。
   * @param[in] contours 输入轮廓集合。
   * @param[in] hierarchy 轮廓层级信息。
   * @param[out] image_corners 输出排序后的四角点。
   * @return 成功返回true；否则返回false。
   */
  bool select_target_corners(
    const std::vector<std::vector<cv::Point>> & contours,
    const std::vector<cv::Vec4i> & hierarchy,
    std::vector<cv::Point2f> & image_corners,
    DebugFrame * debug_frame) const;

  /**
   * @brief 提取并排序轮廓四角。
   * @param[in] contour 输入轮廓。
   * @param[out] ordered_corners 顺时针排序的四角点。
   * @return 成功返回true；否则返回false。
   */
  bool extract_ordered_corners(
    const std::vector<cv::Point> & contour,
    std::vector<cv::Point2f> & ordered_corners) const;

  /**
   * @brief 链码规则筛选候选轮廓。
   * @param[in] contour 输入轮廓。
   * @return 满足链码跳变阈值时返回true。
   */
  bool is_chain_code_candidate(const std::vector<cv::Point> & contour) const;

  /**
   * @brief 根据四角点执行PnP求解并计算重投影误差。
   * @param[in] image_corners 输入图像四角点。
   * @param[out] pose_solution 输出位姿与重投影误差。
   * @return 成功返回true；失败返回false。
   */
  bool solve_target_pose_pnp(
    const std::vector<cv::Point2f> & image_corners,
    PoseSolution & pose_solution) const;

  /**
   * @brief 对四角点进行顺时针排序并以左上角为起点。
   * @param[in,out] corners 待排序角点。
   */
  static void order_corners_clockwise(std::vector<cv::Point2f> & corners);

  /**
   * @brief 计算轮廓评分，分数越小越优。
   * @param[in] contour 输入轮廓。
   * @param[in] corners 对应角点。
   * @return 轮廓评分。
   */
  static double contour_score(
    const std::vector<cv::Point> & contour,
    const std::vector<cv::Point2f> & corners);

  static double normalize_angle(double angle_deg);

  ProcessingConfig processing_config_;
  CameraModel camera_model_;
};

}  // namespace realsense_subscriber

#endif  // REALSENSE_SUBSCRIBER__VISION_PIPELINE_HPP_
