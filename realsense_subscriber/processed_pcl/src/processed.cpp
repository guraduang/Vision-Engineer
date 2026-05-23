#include <boost/thread/thread.hpp>
#include <chrono>
// #include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include <iostream>
// #include <opencv4/opencv2/opencv.hpp>
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/common/geometry.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include "pcl/conversions.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/boundary.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/narf_descriptor.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/ModelCoefficients.h>
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include <pcl/range_image/range_image.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <visualization_msgs/msg/marker.hpp>
#include "realsense_msgs/msg/target.hpp"

class RealSenseSubscriber : public rclcpp::Node
{
private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscription_;
  // rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker1_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr object_center_publisher_;
  // rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr bottom_right_point_publisher_;
  // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_publisher_;
  rclcpp::Publisher<realsense_msgs::msg::Target>::SharedPtr data_publisher_;
  tf2_ros::TransformBroadcaster br;
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

public:
  RealSenseSubscriber() : Node("realsense2_subscriber"), pointcloud_subscription_(nullptr), marker_publisher_(nullptr),
                          object_center_publisher_(nullptr), br(this), marker1_publisher_(nullptr)
  {
    pointcloud_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "camera/camera/depth/color/points", 10, std::bind(&RealSenseSubscriber::pointcloud_callback, this, std::placeholders::_1));
    // pcl_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("normals_cloud", 10);
    data_publisher_ = this->create_publisher<realsense_msgs::msg::Target>("/tracker/target", 10);
    // image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
    //    "camera/camera/color/image_raw", 10, std::bind(&RealSenseSubscriber::image_callback, this, std::placeholders::_1));
    marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("marker_topic", 10);
    marker1_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("marker1_topic", 10);
    object_center_publisher_ = this->create_publisher<geometry_msgs::msg::PointStamped>("center_topic", 10);
    // bottom_right_point_publisher_ = this->create_publisher<geometry_msgs::msg::PointStamped>("bottom_right_point_", 10);
    // viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("Normal viewer"));
  }

private:
  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg_)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_clouds(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> raw_pointclouds;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> sor_pointclouds;
    pcl::fromROSMsg(*pointcloud_msg_, *pcl_cloud); // 无组织点云
    if (Filter(pcl_cloud, raw_pointclouds, sor_pointclouds))
    {
      std::vector<Eigen::Vector4f> center_points;
      std::vector<Eigen::Vector3f> average_normals;
      for (size_t i = 0; i < raw_pointclouds.size(); i++)
      {
        Eigen::Vector4f center_point;
        Eigen::Vector3f average_normal;
        pcl::compute3DCentroid(*(raw_pointclouds[i]), center_point);
        average_normal = compute_averagenormals(sor_pointclouds[i]);
        double cube_size = 0.288;
        double offset_x = center_point(0) - (average_normal(0) * (cube_size / 2));
        double offset_y = center_point(1) - (average_normal(1) * (cube_size / 2));
        double offset_z = center_point(2) - (average_normal(2) * (cube_size / 2));
        average_normals.push_back(average_normal);
        center_points.push_back(center_point);
        *pcl_clouds += *(raw_pointclouds[i]);
      }
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "camera_depth_optical_frame";
      marker.header.stamp = now();
      marker.ns = "filtered_clouds";
      marker.id = 0;
      marker.type = visualization_msgs::msg::Marker::POINTS;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.005; // 点的大小
      marker.scale.y = 0.005;
      marker.color.a = 1.0; // 完全不透明
      marker.color.r = 1.0; // 颜色，这里设置为红色
      for (const auto &point : pcl_clouds->points)
      {
        geometry_msgs::msg::Point p;
        p.x = point.x;
        p.y = point.y;
        p.z = point.z;
        marker.points.push_back(p);
      }
      marker_publisher_->publish(marker);
    }
    // 计算法线和中心点

    // // 计算&&判断
    // // publish_centerpoint(average_normal, output_cloud);
    // // 发布点云
    // visualization_msgs::msg::Marker marker;
    // marker.header.frame_id = "camera_depth_optical_frame";
    // marker.header.stamp = now();
    // marker.ns = "filtered_clouds";
    // marker.id = 0;
    // marker.type = visualization_msgs::msg::Marker::POINTS;
    // marker.action = visualization_msgs::msg::Marker::ADD;
    // marker.pose.orientation.w = 1.0;
    // marker.scale.x = 0.005; // 点的大小
    // marker.scale.y = 0.005;
    // marker.color.a = 1.0; // 完全不透明
    // marker.color.r = 1.0; // 颜色，这里设置为红色
    // for (const auto &point : output_cloud->points)
    // {
    //   geometry_msgs::msg::Point p;
    //   p.x = point.x;
    //   p.y = point.y;
    //   p.z = point.z;
    //   marker.points.push_back(p);
    // }
    // marker_publisher_->publish(marker);
    // vis_nomral(output_cloud);
  }

  bool Filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &raw_pointclouds,
              std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &sor_pointclouds)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>); // 点云缩小范围
    std::vector<pcl::PointIndices> cluster_indices;                                         // 聚类索引
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);   // 直通滤波后的树
    pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZ>); // 条件滤波
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;                                      // 聚类
    pcl::ConditionalRemoval<pcl::PointXYZ> condition;                                       // 条件滤波
    pcl::VoxelGrid<pcl::PointXYZ> vox;                                                      // 下采样
    // 条件直通滤波
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::GT, -0.3)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::LT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::GT, -0.2)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::LT, 0.2)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::LT, 1.2)));
    // 应用条件
    condition.setInputCloud(pcl_cloud);
    condition.setCondition(range_cond);
    condition.filter(*cloud_filtered);
    // 下采样
    vox.setInputCloud(cloud_filtered);
    vox.setLeafSize(0.01f, 0.01f, 0.01f);
    vox.filter(*cloud_filtered);
    if (cloud_filtered->size() > 1000)
    {
      // 聚类
      tree->setInputCloud(cloud_filtered);
      ec.setClusterTolerance(0.015f); // 1.2cm
      ec.setMinClusterSize(800);
      ec.setSearchMethod(tree);
      ec.setInputCloud(cloud_filtered);
      ec.extract(cluster_indices);
      for (const auto &cluster : cluster_indices) // 外面一层是聚类的数量
      {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);

        for (const auto &idx : cluster.indices) // 这一层是每个聚类的组成的点
        {
          cloud_cluster->push_back((*cloud_filtered)[idx]);
        }
        std::cout << "聚类的点的大小：" << cloud_cluster->size() << std::endl;
        plane(cloud_cluster, raw_pointclouds, sor_pointclouds);
        if (raw_pointclouds.size() > 0 && sor_pointclouds.size() > 0 && raw_pointclouds.size() == sor_pointclouds.size())
        {
          return raw_pointclouds.size();
          break;
        }
      }
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "Point Cloud is not enough!");
      return 0;
    }

    // visualization_msgs::msg::Marker marker1;
    // marker1.header.frame_id = "camera_depth_optical_frame";
    // marker1.header.stamp = now();
    // marker1.ns = "filtered1_clouds";
    // marker1.id = 0;
    // marker1.type = visualization_msgs::msg::Marker::POINTS;
    // marker1.action = visualization_msgs::msg::Marker::ADD;
    // marker1.pose.orientation.w = 1.0;
    // marker1.scale.x = 0.005; // 点的大小
    // marker1.scale.y = 0.005;
    // marker1.color.a = 1.0; // 完全不透明
    // marker1.color.r = 1.0; // 颜色，这里设置为红色
    // for (const auto &point : cloud_cluster->points)
    // {
    //   geometry_msgs::msg::Point p;
    //   p.x = point.x;
    //   p.y = point.y;
    //   p.z = point.z;
    //   marker1.points.push_back(p);
    // }
    // marker1_publisher_->publish(marker1);
  }

  // 获取聚类的所有面
  void plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_cluster, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &raw_pointclouds,
             std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &sor_pointclouds)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZ>(*cloud_cluster));
    // 设定平面分割参数
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(500);
    seg.setDistanceThreshold(0.008f); // 设定距离阈值，可以根据需要调整
    // 循环执行平面分割
    while (remaining_cloud->size() > 0)
    {
      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
      // 执行平面分割
      seg.setInputCloud(remaining_cloud);
      seg.segment(*inliers, *coefficients);
      // 如果没有找到平面，结束循环
      if (inliers->indices.size() < 300)
      {
        break;
      }
      // 创建一个新的平面点云
      pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cluster(new pcl::PointCloud<pcl::PointXYZ>);
      // 提取平面点
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(remaining_cloud);
      extract.setIndices(inliers);
      extract.filter(*plane_cluster);
      // 将当前平面点云添加到向量中
      if (isRightAspectRatio(plane_cluster))
      {
        // 平滑
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
        sor.setInputCloud(plane_cluster);
        sor.setMeanK(0.6 * plane_cluster->size()); // 邻域点的个数，越大越平滑，但可能丢失细节
        sor.setStddevMulThresh(1.0);               // 标准差的倍数，根据实际情况调整
        sor.filter(*temp_pointcloud);
        sor_pointclouds.push_back(temp_pointcloud);
        raw_pointclouds.push_back(plane_cluster);
      }
      // 从原始点云中移除已经提取的平面点
      pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud_after_extraction(new pcl::PointCloud<pcl::PointXYZ>);
      extract.setNegative(true);
      extract.filter(*remaining_cloud_after_extraction);
      remaining_cloud = remaining_cloud_after_extraction;
    }
    std::cout << "平面数量：" << sor_pointclouds.size() << std::endl;
  }

  // 可视化向量
  void vis_nomral(const pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud)
  {
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normals = compute_normals(output_cloud);
    viewer->removeAllPointClouds();
    viewer->removeAllShapes();
    //  设置背景颜色
    viewer->setBackgroundColor(0.3, 0.3, 0.3);
    viewer->addText("normal", 10, 10, "text");
    // 设置点云颜色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(output_cloud, 255, 0, 0);
    // 添加坐标系
    viewer->addPointCloud<pcl::PointXYZ>(output_cloud, single_color, "sample cloud");
    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(output_cloud, normals, 1, 0.01, "normals");
    // 设置点云大小
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // 复制点云
    pcl::copyPointCloud(*output_cloud, *cloud);
    viewer->spinOnce();
  }

  // 发布中心点坐标
  void publish_centerpoint(const Eigen::Vector3f normals, const pcl::PointCloud<pcl::PointXYZ>::Ptr input_points)
  {
    // 计算中心点
    Eigen::Vector4d centroid_point;
    pcl::compute3DCentroid(*input_points, centroid_point);
    double cube_size = 0.288;
    double offset_x = centroid_point(0) - (normals(0) * (cube_size / 2));
    double offset_y = centroid_point(1) - (normals(1) * (cube_size / 2));
    double offset_z = centroid_point(2) - (normals(2) * (cube_size / 2));
    geometry_msgs::msg::PointStamped cube_center_msg;
    cube_center_msg.header.stamp = this->now();
    cube_center_msg.header.frame_id = "camera_depth_optical_frame";
    cube_center_msg.point.x = offset_x;
    cube_center_msg.point.y = offset_y;
    cube_center_msg.point.z = offset_z;
    object_center_publisher_->publish(cube_center_msg);
    // Eigen::Vector3d centerPoint_(offset_x, offset_y, offset_z);
    // publishTF("camera_depth_optical_frame", "rectface", centerPoint_, average_normal);
  }

  // 计算平面长宽比
  double isRightAspectRatio(const pcl::PointCloud<pcl::PointXYZ>::Ptr input_points)
  {
    if (input_points->empty())
    {
      std::cerr << "Error: Input point cloud is empty." << std::endl;
      return 0;
    }
    // Compute the bounding box of the plane
    pcl::PointXYZ min_point, max_point;
    pcl::getMinMax3D(*input_points, min_point, max_point);
    // Calculate the dimensions of the bounding box
    double length = std::abs(max_point.x - min_point.x);
    double width = std::abs(max_point.y - min_point.y);
    // Calculate the aspect ratio
    double aspect_ratio = std::abs(length / width);
    if (aspect_ratio > 0.70 && aspect_ratio < 1.30)
    {
      std::cout << "长宽比：" << aspect_ratio << std::endl;
      return aspect_ratio;
    }
    else
    {
      return 0;
    }
  }

  // 计算边缘点
  void edge_detection(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_ptr, pcl::PointCloud<pcl::Boundary>::Ptr &output)
  {
    pcl::PointCloud<pcl::Normal>::Ptr normal_ptr(new pcl::PointCloud<pcl::Normal>);
    normal_ptr = this->compute_normals(cloud_ptr);
    std::vector<int> neighbor_idx;
    std::vector<float> neighbor_dist;
    pcl::search::KdTree<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_ptr);
    Eigen::Vector4f u = Eigen::Vector4f::Zero(), v = Eigen::Vector4f::Zero();
    output->resize(cloud_ptr->size());
    int boundary_size = 0;

    for (size_t i = 0; i < cloud_ptr->size(); i++)
    {
      kdtree.nearestKSearch(cloud_ptr->points[i], 40, neighbor_idx, neighbor_dist);
      this->getCoordinateSystemOnPlane(normal_ptr->points[i], u, v);
      output->points[i].boundary_point = this->isBoundaryPoint(cloud_ptr, cloud_ptr->points[i], neighbor_idx, u, v, 0.9);
    }
  }

  // 计算点云的面的所有法线
  pcl::PointCloud<pcl::Normal>::Ptr compute_normals(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_ptr)
  {
    pcl::PointCloud<pcl::Normal>::Ptr normal_ptr(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud_ptr);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(kdtree);
    ne.setKSearch(30);
    ne.compute(*normal_ptr);
    return normal_ptr;
  }

  // 计算归一化后一整个面的法线
  Eigen::Vector3f compute_averagenormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_ptr)
  {
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud_ptr);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(kdtree);
    ne.setKSearch(30);
    ne.compute(*normals);
    Eigen::Vector3f average_normal(0.0, 0.0, 0.0);
    for (size_t i = 0; i < normals->size(); ++i)
    {
      average_normal += Eigen::Vector3f(normals->points[i].normal_x,
                                        normals->points[i].normal_y,
                                        normals->points[i].normal_z);
    }
    average_normal /= normals->size();
    average_normal.normalize();
    std::cout << "X: " << average_normal(0) << " Y: " << average_normal(1) << " Z: " << average_normal(2) << std::endl;
    return average_normal;
  }

  void getCoordinateSystemOnPlane(const pcl::Normal &p_coeff, Eigen::Vector4f &u, Eigen::Vector4f &v)
  {
    pcl::Vector4fMapConst p_coeff_v = p_coeff.getNormalVector4fMap();
    v = p_coeff_v.unitOrthogonal();
    u = p_coeff_v.cross3(v);
  }

  bool isBoundaryPoint(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const pcl::PointXYZ &q_point,
                       const std::vector<int> &indices, const Eigen::Vector4f &u, const Eigen::Vector4f &v, const float angle_threshold)
  {
    if (indices.size() < 3)
      return false;
    if (!std::isfinite(q_point.x) || !std::isfinite(q_point.y) || !std::isfinite(q_point.z))
      return false;
    std::vector<float> angles(indices.size());
    float max_dif = FLT_MIN, dif;
    int cp = 0;
    for (const auto &index : indices)
    {
      if (!std::isfinite(cloud->points[index].x) ||
          !std::isfinite(cloud->points[index].y) ||
          !std::isfinite(cloud->points[index].z))
        continue;
      Eigen::Vector4f delta = cloud->points[index].getVector4fMap() - q_point.getVector4fMap();
      if (delta == Eigen::Vector4f::Zero())
        continue;
      angles[cp++] = std::atan2(v.dot(delta), u.dot(delta));
    }
    if (cp == 0)
      return false;
    angles.resize(cp);
    std::sort(angles.begin(), angles.end());
    for (size_t i = 0; i < angles.size(); i++)
    {
      dif = angles[i + 1] - angles[i];
      max_dif = std::max(max_dif, dif);
    }
    dif = 2 * static_cast<float>(M_PI) - angles.back() + angles[0];
    max_dif = std::max(max_dif, dif);
    return (max_dif > angle_threshold);
  }
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto rs_subscriber = std::make_shared<RealSenseSubscriber>();
  rclcpp::spin(rs_subscriber);
  rclcpp::shutdown();
  return 0;
}
