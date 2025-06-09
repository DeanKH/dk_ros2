#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/PolygonMesh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/surface/poisson.h>

#include <Eigen/Core>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cmath>
#include <dk_perception/detection/d3/rectangle_detection.hpp>
#include <dk_perception/dnn/superpoint.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "dk_perception/detection/d2/node_extractor.hpp"
#include "dk_perception/type/pointcloud/rgbd_type.hpp"

template <typename ImageType>
void minmaxFilter(cv::Mat& img, ImageType min, ImageType max) {
  cv::Mat mask = (img >= min) & (img <= max);
  img.setTo(0, ~mask);  // Set pixels outside the range to 0
}

pcl::PolygonMesh generateMesh(
    const pcl::PointCloud<pcl::PointNormal>::Ptr& cloud) {
  pcl::Poisson<pcl::PointNormal> poisson;
  int poisson_depth = 8;
  poisson.setDepth(poisson_depth);
  poisson.setInputCloud(cloud);

  pcl::PolygonMesh mesh;
  poisson.reconstruct(mesh);
  return mesh;
}

using namespace Eigen;

// TODO: 高速化のためにhttps://github.com/lighttransport/nanortに置き換える
bool rayTriangleIntersect(const Vector3f& orig, const Vector3f& dir,
                          const Vector3f& v0, const Vector3f& v1,
                          const Vector3f& v2, float& t, float& u, float& v) {
  const float EPSILON = 1e-7;
  Vector3f edge1 = v1 - v0;
  Vector3f edge2 = v2 - v0;
  Vector3f h = dir.cross(edge2);
  float a = edge1.dot(h);
  if (fabs(a) < EPSILON) return false;  // parallel

  float f = 1.0 / a;
  Vector3f s = orig - v0;
  u = f * s.dot(h);
  if (u < 0.0 || u > 1.0) return false;

  Vector3f q = s.cross(edge1);
  v = f * dir.dot(q);
  if (v < 0.0 || u + v > 1.0) return false;

  t = f * edge2.dot(q);
  if (t > EPSILON) return true;  // hit

  return false;
}

class Rectangle3dDetectionNode : public rclcpp::Node {
 public:
  explicit Rectangle3dDetectionNode(const rclcpp::NodeOptions& options)
      : Node("rectangle3d_detection_node", options) {
    RCLCPP_DEBUG(this->get_logger(), "Starting Rectangle3D Detection Node");

    // Declare parameters
    this->declare_parameter("use_exact_time", false);
    this->declare_parameter("queue_size", 10);
    this->declare_parameter("depth_factor", 1000.0);
    this->declare_parameter("depth_min_threshold", 0.75);
    this->declare_parameter("depth_max_threshold", 2.0);

    this->declare_parameter("superpoint_vocab_dict_path",
                            "focus_descriptors.csv");
    this->declare_parameter("superpoint_model_path", "superpoint.onnx");
    this->declare_parameter("superpoint_vocab_similarity_threshold", 0.5);
    this->declare_parameter("superpoint_min_score", 0.0005);

    this->declare_parameter("voxel_leaf_size", 0.01);
    // Get parameters

    superpoint_vocab_dict_path_ =
        this->get_parameter("superpoint_vocab_dict_path").as_string();

    if (!std::filesystem::exists(superpoint_vocab_dict_path_)) {
      RCLCPP_FATAL(this->get_logger(),
                   "SuperPoint vocabulary dictionary path does not exist: %s",
                   superpoint_vocab_dict_path_.c_str());
      rclcpp::shutdown();
    }

    std::string superpoint_model_path =
        this->get_parameter("superpoint_model_path").as_string();
    if (superpoint_model_path.empty()) {
      RCLCPP_FATAL(this->get_logger(), "SuperPoint model path is not set");
      rclcpp::shutdown();
    }

    superpoint_ = std::make_shared<dklib::experimental::SuperPoint>(
        superpoint_model_path,
        dklib::experimental::SuperPoint::InferenceDevice::kCUDA,
        dklib::experimental::SuperPoint::InputSize::kInputSize1024);

    bool use_exact_time = this->get_parameter("use_exact_time").as_bool();
    int queue_size = this->get_parameter("queue_size").as_int();
    depth_factor_ = this->get_parameter("depth_factor").as_double();
    depth_min_threshold_ =
        this->get_parameter("depth_min_threshold").as_double();
    depth_max_threshold_ =
        this->get_parameter("depth_max_threshold").as_double();

    process_image_publisher_ =
        this->create_publisher<sensor_msgs::msg::Image>("processed_image", 10);
    mesh_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "mesh_visualization", 10);
    mesh_array_publisher_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "mesh_array_visualization", 10);

    // Initialize subscribers with message filters
    image1_sub_.subscribe(this, "color_image");
    image2_sub_.subscribe(this, "depth_image");
    camera_info_sub_.subscribe(this, "rgbd_camera_info");

    // Set up synchronization policy
    if (use_exact_time) {
      // Use exact time synchronization
      exact_policy_ = std::make_shared<ExactSyncPolicy>(queue_size);
      exact_sync_ = std::make_shared<ExactSync>(*exact_policy_);
      exact_sync_->connectInput(image1_sub_, image2_sub_, camera_info_sub_);
      exact_sync_->registerCallback(std::bind(
          &Rectangle3dDetectionNode::callback, this, std::placeholders::_1,
          std::placeholders::_2, std::placeholders::_3));

      RCLCPP_DEBUG(this->get_logger(),
                   "Using Exact Time synchronization policy");
    } else {
      // Use approximate time synchronization (default)
      approx_policy_ = std::make_shared<ApproxSyncPolicy>(queue_size);
      approx_sync_ = std::make_shared<ApproxSync>(*approx_policy_);
      approx_sync_->connectInput(image1_sub_, image2_sub_, camera_info_sub_);
      approx_sync_->registerCallback(std::bind(
          &Rectangle3dDetectionNode::callback, this, std::placeholders::_1,
          std::placeholders::_2, std::placeholders::_3));

      RCLCPP_DEBUG(this->get_logger(),
                   "Using Approximate Time synchronization policy");
    }
  }

 private:
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
      process_image_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mesh_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      mesh_array_publisher_;

  using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image,
      sensor_msgs::msg::CameraInfo>;
  using ExactSyncPolicy =
      message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image,
                                                sensor_msgs::msg::Image,
                                                sensor_msgs::msg::CameraInfo>;
  using ApproxSync = message_filters::Synchronizer<ApproxSyncPolicy>;
  using ExactSync = message_filters::Synchronizer<ExactSyncPolicy>;

  // Subscribers
  message_filters::Subscriber<sensor_msgs::msg::Image> image1_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> image2_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;

  // Synchronization policies
  std::shared_ptr<ApproxSyncPolicy> approx_policy_;
  std::shared_ptr<ExactSyncPolicy> exact_policy_;
  std::shared_ptr<ApproxSync> approx_sync_;
  std::shared_ptr<ExactSync> exact_sync_;

  double depth_factor_;
  double depth_min_threshold_;
  double depth_max_threshold_;
  std::string superpoint_vocab_dict_path_;
  std::shared_ptr<dklib::experimental::SuperPoint> superpoint_;

  // Callback function for synchronized messages
  void callback(
      const sensor_msgs::msg::Image::ConstSharedPtr& image1_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr& image2_msg,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg) {
    RCLCPP_DEBUG(this->get_logger(), "Received synchronized messages");
    RCLCPP_DEBUG(this->get_logger(), "Image1 timestamp: %u.%u",
                 image1_msg->header.stamp.sec,
                 image1_msg->header.stamp.nanosec);
    RCLCPP_DEBUG(this->get_logger(), "Image2 timestamp: %u.%u",
                 image2_msg->header.stamp.sec,
                 image2_msg->header.stamp.nanosec);
    RCLCPP_DEBUG(this->get_logger(), "CameraInfo timestamp: %u.%u",
                 camera_info_msg->header.stamp.sec,
                 camera_info_msg->header.stamp.nanosec);

    try {
      // Convert ROS image messages to OpenCV images
      cv_bridge::CvImageConstPtr cv_image1 =
          cv_bridge::toCvShare(image1_msg, "bgr8");
      cv_bridge::CvImageConstPtr cv_image2 = cv_bridge::toCvShare(image2_msg);
      Eigen::Matrix3f intrinsic =
          Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
              camera_info_msg->k.data())
              .cast<float>();

      cv::Mat depth_image = cv_image2->image;

      minmaxFilter<uint16_t>(depth_image,
                             std::lround(depth_min_threshold_ * depth_factor_),
                             std::lround(depth_max_threshold_ * depth_factor_));

      dklib::perception::type::pointcloud::DepthImageSet rgbd{
          cv_image1->image, depth_image, intrinsic,
          static_cast<float>(1.0 / depth_factor_)};

      // Process images and camera info
      processImages(rgbd, image1_msg->header);
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
    }
  }

  // Process the synchronized images and camera info
  void processImages(
      const dklib::perception::type::pointcloud::DepthImageSet& rgbd,
      const std_msgs::msg::Header& header) {
    const double superpoint_vocab_similarity_threshold =
        this->get_parameter("superpoint_vocab_similarity_threshold")
            .as_double();
    const double superpoint_min_score =
        this->get_parameter("superpoint_min_score").as_double();

    dklib::perception::detection::d2::NodeExtractorFromImageFeatures
        node_extractor{superpoint_, superpoint_vocab_dict_path_};
    node_extractor.setSimilarityThreshold(
        superpoint_vocab_similarity_threshold);
    node_extractor.setMinScore(superpoint_min_score);
    node_extractor.setROI(cv::Rect(100, 100, rgbd.color_image().cols - 200,
                                   rgbd.color_image().rows - 200));
    auto points = node_extractor.extract(rgbd.color_image());
    RCLCPP_DEBUG_STREAM(this->get_logger(),
                        "Extracted " << points.size() << " keypoints");
    cv::Mat draw_image = rgbd.color_image().clone();
    for (const auto& point : points) {
      cv::circle(draw_image, point, 3, cv::Scalar(0, 255, 0), -1);
    }

    {
      auto msg = cv_bridge::CvImage(header, "bgr8", draw_image).toImageMsg();
      process_image_publisher_->publish(*msg);
    }
    // 3d
    dklib::perception::type::pointcloud::
        IteratableColorizedPointCloudReadOnlyAccessor<
            dklib::perception::type::pointcloud::DepthImageSet>
            accessor(rgbd);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(
        new pcl::PointCloud<pcl::PointNormal>);

    cloud->reserve(accessor.size());
    for (size_t i = 0; i < accessor.size(); ++i) {
      const Eigen::Vector3f point = accessor.point_at(i);
      cloud->emplace_back(point.x(), point.y(), point.z());
    }
    RCLCPP_INFO(this->get_logger(), "Point cloud size: %zu", cloud->size());
    // Remove any NaN points
    std::vector<int> indices;
    cloud->is_dense = false;  // Set to false to allow NaN points
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    pcl::VoxelGrid<pcl::PointNormal> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    const float leaf_size =
        static_cast<float>(this->get_parameter("voxel_leaf_size").as_double());
    voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_grid.filter(*cloud);
    RCLCPP_INFO(this->get_logger(), "After voxel grid filtering: %zu points",
                cloud->size());
    pcl::StatisticalOutlierRemoval<pcl::PointNormal> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud);

    pcl::search::KdTree<pcl::PointNormal>::Ptr tree(
        new pcl::search::KdTree<pcl::PointNormal>());
    tree->setInputCloud(cloud);

    auto ne_start_time = std::chrono::high_resolution_clock::now();
    pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(20);
    ne.compute(*cloud);
    auto ne_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> ne_duration = ne_end_time - ne_start_time;
    RCLCPP_INFO(this->get_logger(), "Normal estimation took %.3f seconds",
                ne_duration.count());

    auto mesh_reconstruct_start_time =
        std::chrono::high_resolution_clock::now();
    pcl::PolygonMesh mesh = generateMesh(cloud);
    auto mesh_reconstruct_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> mesh_reconstruct_duration =
        mesh_reconstruct_end_time - mesh_reconstruct_start_time;
    RCLCPP_INFO(this->get_logger(), "Mesh reconstruction took %.3f seconds",
                mesh_reconstruct_duration.count());

    pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.cloud, *mesh_cloud);

    Eigen::Matrix3f intrinsic = rgbd.intrinsic();

    auto intersection_start_time = std::chrono::high_resolution_clock::now();
    Eigen::Vector3f ray_origin(0.0f, 0.0f, 0.0f);
    std::vector<Eigen::Vector3f> hit_points;
    for (const auto& pt : points) {
      Eigen::Vector3f ray_direction((pt.x - intrinsic(0, 2)) / intrinsic(0, 0),
                                    (pt.y - intrinsic(1, 2)) / intrinsic(1, 1),
                                    1.0f);
      ray_direction.normalize();

      float closest_t = std::numeric_limits<float>::infinity();
      Eigen::Vector3f hit_point;
      for (const auto& polygon : mesh.polygons) {
        if (polygon.vertices.size() < 3) continue;  // Skip invalid polygons
        const pcl::PointXYZ& v0 = mesh_cloud->points[polygon.vertices[0]];
        const pcl::PointXYZ& v1 = mesh_cloud->points[polygon.vertices[1]];
        const pcl::PointXYZ& v2 = mesh_cloud->points[polygon.vertices[2]];
        float t, u, v;
        bool hit = rayTriangleIntersect(
            ray_origin, ray_direction, Eigen::Vector3f(v0.x, v0.y, v0.z),
            Eigen::Vector3f(v1.x, v1.y, v1.z),
            Eigen::Vector3f(v2.x, v2.y, v2.z), t, u, v);
        if (hit) {
          if (t > 0 && t < closest_t) {
            closest_t = t;
            hit_point = ray_origin + t * ray_direction;
          }
        }
      }

      if (std::isfinite(closest_t) &&
          closest_t < std::numeric_limits<float>::infinity()) {
        pcl::PointNormal search_point(hit_point.x(), hit_point.y(),
                                      hit_point.z());
        std::vector<int> k_indices;
        std::vector<float> k_sqr_distances;
        tree->nearestKSearch(search_point, 1, k_indices, k_sqr_distances);
        constexpr double distance_threshold = 0.03;
        if (k_sqr_distances.empty() ||
            k_sqr_distances[0] > distance_threshold * distance_threshold) {
          continue;
        }
        hit_points.push_back(hit_point);
      }
    }
    auto intersection_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> intersection_duration =

        intersection_end_time - intersection_start_time;
    RCLCPP_INFO(this->get_logger(),
                "intersection mesh and rays took %.3f seconds",
                intersection_duration.count());

    dklib::perception::detection::d3::RightAngleTriangleConstructor<
        Eigen::Vector3f>
        rat_constructor;

    auto rat_start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::array<size_t, 3>> triangles =
        rat_constructor.construct(hit_points);
    auto rat_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> rat_duration = rat_end_time - rat_start_time;
    RCLCPP_INFO(this->get_logger(),
                "Right-angle triangle construction took %.3f seconds",
                rat_duration.count());

    publishMeshMarker(mesh, header);
    publishPointsMarker(hit_points, header);
    publishTrianglesMarker(hit_points, triangles, header);

    dklib::perception::detection::d3::RectangleDetection<
        pcl::Poisson<pcl::PointNormal>>
        detector;
  }

  void publishPointsMarker(const std::vector<Eigen::Vector3f>& points,
                           const std_msgs::msg::Header& header) {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "points";
    marker.id = 1;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.01;  // Point size
    marker.scale.y = 0.01;  // Point size
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    marker.points.reserve(points.size());
    for (const auto& point : points) {
      geometry_msgs::msg::Point p;
      p.x = point.x();
      p.y = point.y();
      p.z = point.z();
      marker.points.push_back(p);
    }

    mesh_publisher_->publish(marker);
    RCLCPP_DEBUG(this->get_logger(), "Points visualization published.");
  }

  void publishTrianglesMarker(
      const std::vector<Eigen::Vector3f>& points,
      const std::vector<std::array<size_t, 3>>& triangles,
      const std_msgs::msg::Header& header)  // NOLINT
  {
    visualization_msgs::msg::MarkerArray marker_array;
    for (size_t i = 0; i < triangles.size(); ++i) {
      if (triangles[i].size() != 3) {
        RCLCPP_WARN(this->get_logger(),
                    "Triangle %zu does not have exactly 3 vertices, skipping.",
                    i);
        continue;
      }

      visualization_msgs::msg::Marker marker;
      marker.header = header;
      marker.ns = "triangles";
      marker.id = static_cast<uint32_t>(i) + 2;
      marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = 0;
      marker.pose.position.y = 0;
      marker.pose.position.z = 0;
      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.005;
      marker.scale.y = 1.0;
      marker.scale.z = 1.0;
      marker.color.r = 0.0;
      marker.color.g = 0.0;
      marker.color.b = 1.0;
      marker.color.a = 1.0;

      for (size_t j = 0; j < 3; ++j) {
        if (triangles[i][j] >= points.size()) {
          RCLCPP_WARN(this->get_logger(),
                      "Triangle %zu has vertex index %zu out of bounds, "
                      "skipping.",
                      i, triangles[i][j]);
          continue;
        }
        geometry_msgs::msg::Point start_point;
        start_point.x = points[triangles[i][j]].x();
        start_point.y = points[triangles[i][j]].y();
        start_point.z = points[triangles[i][j]].z();

        geometry_msgs::msg::Point end_point;
        end_point.x = points[triangles[i][(j + 1) % 3]].x();
        end_point.y = points[triangles[i][(j + 1) % 3]].y();
        end_point.z = points[triangles[i][(j + 1) % 3]].z();
        marker.points.push_back(start_point);
        marker.points.push_back(end_point);
      }
      marker_array.markers.push_back(marker);
    }

    mesh_array_publisher_->publish(marker_array);
  }

  void publishMeshMarker(const pcl::PolygonMesh& mesh,
                         const std_msgs::msg::Header& header) {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "mesh";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;
    marker.frame_locked = false;

    // Convert the mesh to triangle list
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);

    // Add all triangles to the marker
    for (const auto& polygon : mesh.polygons) {
      if (polygon.vertices.size() >= 3) {
        for (size_t i = 0; i < 3; ++i) {
          geometry_msgs::msg::Point p;
          p.x = cloud.points[polygon.vertices[i]].x;
          p.y = cloud.points[polygon.vertices[i]].y;
          p.z = cloud.points[polygon.vertices[i]].z;
          marker.points.push_back(p);
        }
      }
    }

    mesh_publisher_->publish(marker);
    RCLCPP_DEBUG(this->get_logger(),
                 "Mesh visualization published with %zu points.",
                 marker.points.size());
  }
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto node = std::make_shared<Rectangle3dDetectionNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
