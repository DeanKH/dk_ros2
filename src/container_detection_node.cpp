#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <voxblox/mesh/mesh_integrator.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <dk_perception/detection/d3/marker_target_detection.hpp>
#include <dk_perception/geometry/bounding_box_3d.hpp>
#include <dk_perception/optimization/placement_pose_esdf_based_optimizer.hpp>
#include <dk_perception/reconstruction/reconstruction.hpp>
#include <dk_perception/rerun/publish_data.hpp>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rerun.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "dk_perception/detection/d3/radial_extremum_detector.hpp"
#include "dk_perception/type/pointcloud/iteratable_colorized_pointcloud_accessor.hpp"
#include "dk_perception/type/pointcloud/rgbd_type.hpp"
#include "dk_ros2/converter.hpp"

template <typename ImageType>
void minmaxFilter(cv::Mat& img, ImageType min, ImageType max) {
  cv::Mat mask = (img >= min) & (img <= max);
  img.setTo(0, ~mask);  // Set pixels outside the range to 0
}

class RGBDProcessNode : public rclcpp::Node {
 public:
  explicit RGBDProcessNode(const rclcpp::NodeOptions& options)
      : Node("rgbd_process_node", options) {
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "container_area", 10);
    box_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "container_box", 10);

    pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "container_pointcloud", 10);
    rec_ = std::make_shared<rerun::RecordingStream>("rgbd_process_node");
    try {
      rec_->spawn().throw_on_failure();
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to spawn Rerun recording stream: %s", e.what());
      rec_ = nullptr;
    }

    // Declare parameters for crop box filter
    this->declare_parameter("crop_box.min_x", -1.0);
    this->declare_parameter("crop_box.min_y", -1.0);
    this->declare_parameter("crop_box.min_z", -1.0);
    this->declare_parameter("crop_box.max_x", 1.0);
    this->declare_parameter("crop_box.max_y", 1.0);
    this->declare_parameter("crop_box.max_z", 1.0);

    // Initialize subscribers with message filters
    image1_sub_.subscribe(this, "color_image");
    image2_sub_.subscribe(this, "depth_image");
    camera_info_sub_.subscribe(this, "rgbd_camera_info");

    // Set up synchronization policy
    if (use_exact_time_) {
      // Use exact time synchronization
      exact_policy_ = std::make_shared<ExactSyncPolicy>(queue_size_);
      exact_sync_ = std::make_shared<ExactSync>(*exact_policy_);
      exact_sync_->connectInput(image1_sub_, image2_sub_, camera_info_sub_);
      exact_sync_->registerCallback(
          std::bind(&RGBDProcessNode::callback, this, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3));

      RCLCPP_DEBUG(this->get_logger(),
                   "Using Exact Time synchronization policy");
    } else {
      // Use approximate time synchronization (default)
      approx_policy_ = std::make_shared<ApproxSyncPolicy>(queue_size_);
      approx_sync_ = std::make_shared<ApproxSync>(*approx_policy_);
      approx_sync_->connectInput(image1_sub_, image2_sub_, camera_info_sub_);
      approx_sync_->registerCallback(
          std::bind(&RGBDProcessNode::callback, this, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3));

      RCLCPP_DEBUG(this->get_logger(),
                   "Using Approximate Time synchronization policy");
    }
  }

  // Callback function for synchronized messages
  void callbackBoxDetection(
      const sensor_msgs::msg::Image::ConstSharedPtr& image1_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr& image2_msg,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg) {
    std::cout << "----------------------------------" << std::endl;
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

    //  measure processing time
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
      publishDepthImageData(rec_, "/marker_47/camera/image/depth", depth_image,
                            depth_factor_);

      dklib::perception::type::pointcloud::DepthImageSet rgbd{
          cv_image1->image, depth_image, intrinsic,
          static_cast<float>(1.0 / depth_factor_)};

      dklib::perception::type::pointcloud::
          IteratableColorizedPointCloudReadOnlyAccessor<
              dklib::perception::type::pointcloud::DepthImageSet>
              accessor(rgbd);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>());
      cloud->resize(accessor.size());

      auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
      for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(accessor.size()); ++i) {
        pcl::PointXYZRGB& point = cloud->points[i];
        const auto& pt = accessor.point_at(i);
        point.x = pt[0];
        point.y = pt[1];
        point.z = pt[2];
        auto color = accessor.color_at(i).value();
        point.r = color[2];
        point.g = color[1];
        point.b = color[0];
      }

      // remove nan points
      cloud->is_dense = false;
      {
        std::vector<int> dummy_indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud, dummy_indices);
      }
      std::cout << "Point cloud size before outlier removal: " << cloud->size()
                << std::endl;
      // Voxel grid filter
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(cloud);
      {
        float voxel_size = 0.01f;
        voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
      }
      voxel_filter.filter(*cloud);
      publishData<pcl::PointXYZRGB>(rec_, "/marker_47/camera/points", cloud);

      // Remove outliers
      pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
      sor.setInputCloud(cloud);
      sor.setMeanK(30);
      sor.setStddevMulThresh(1.0);
      sor.filter(*cloud);

      std::cout << "Point cloud size after outlier removal: " << cloud->size()
                << std::endl;

      const Eigen::Vector3f origin = [&cloud]() {
        return Eigen::Vector3f::Zero();
      }();

      std::cout << "Point cloud origin: [" << origin.transpose() << "]"
                << std::endl;

      dklib::perception::pcproc::RadialSplitter<pcl::PointXYZRGB> splitter;
      splitter.setInputCloud(cloud);
      splitter.setCenter(origin);
      splitter.setAxis(Eigen::Vector3f::UnitZ());
      splitter.setAngleStep(5.0f);
      splitter.setWidth(0.01f);

      dklib::perception::detection::d3::RadialExtremumDetector<pcl::PointXYZRGB>
          detector(splitter);
      std::cout << "Detecting radial segments..." << std::endl;
      auto [bbox, min_points] = detector.execute();
      publishData(rec_, "marker_47/camera/container", bbox);

      std::cout << "Detected " << min_points->size() << " minimum points."
                << std::endl;
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> processing_time = end - start;
      std::cout << "Processing time: " << processing_time.count() << " ms"
                << std::endl;

      double voxel_size = 0.02;
      dklib::perception::reconstruction::BoxInteriorReconstructor reconstructor(
          bbox, voxel_size);
      reconstructor.update<pcl::PointXYZRGB>(*cloud,
                                             Eigen::Matrix4f::Identity());
      pcl::PointCloud<pcl::PointXYZI>::Ptr icloud(
          new pcl::PointCloud<pcl::PointXYZI>());
      *icloud = reconstructor.getSdfVoxelInBox();

      if (rec_) {
        publishData(rec_, "marker_47/camera/container/sdf",
                    Eigen::Isometry3d(bbox.getTransformation().cast<double>()));
        publishVoxelData<pcl::PointXYZI>(
            rec_, "marker_47/camera/container/sdf/tsdf", icloud, voxel_size);

        pcl::PointCloud<pcl::PointXYZI>::Ptr ecloud(
            new pcl::PointCloud<pcl::PointXYZI>());
        *ecloud = reconstructor.getEsdfVoxelInBox();
        publishVoxelData<pcl::PointXYZI>(
            rec_, "marker_47/camera/container/sdf/esdf", ecloud, voxel_size);

        // mesh generation
        {
          auto [cloud, polygons] = reconstructor.generateMesh();
          publishMeshData(rec_, "marker_47/camera/container/sdf/mesh", cloud,
                          polygons);
        }

        dklib::perception::geometry::BoundingBox3D placement_target;
        placement_target.size = Eigen::Vector3d(0.096, 0.063, 0.05);
        placement_target.center = Eigen::Vector3d(0.0, 0.0, 0.0);
        // rotation from box_top coord.
        placement_target.orientation =
            Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());

        dklib::perception::optimization::PlacementPoseEsdfBasedOptimizer
            place_optimizer;

        pcl::PointCloud<pcl::PointXYZI>::Ptr ecloud_filtered(
            new pcl::PointCloud<pcl::PointXYZI>);

        auto optimized_place_box = place_optimizer.optimizePlacementPose(
            placement_target, reconstructor.getEsdfMap(), ecloud_filtered);

        publishVoxelData<pcl::PointXYZI>(
            rec_, "marker_47/camera/container/sdf/placeable", ecloud_filtered,
            voxel_size);

        if (optimized_place_box) {
          Eigen::Vector3d shift_dir;
          auto bottom_grid_points = place_optimizer.getBoxBottomGridPoints(
              *optimized_place_box, reconstructor.getEsdfMap(), shift_dir);
          rerun::publishArrowData(
              rec_, "marker_47/camera/container/sdf/box_bottom_grid/arrow",
              optimized_place_box->center, shift_dir);
          publishVoxelData<pcl::PointXYZI>(
              rec_, "marker_47/camera/container/sdf/box_bottom_grid",
              bottom_grid_points, voxel_size, 1.0f);
          publishData(rec_, "marker_47/camera/container/sdf/placeable",
                      *optimized_place_box, {0, 0, 255, 200}, 0.01f,
                      rerun::components::FillMode::Solid);
          *optimized_place_box =
              place_optimizer.refinePlacementPoseByUniformlyXY(
                  *optimized_place_box, reconstructor.getEsdfMap(), bbox);
          publishData(rec_, "marker_47/camera/container/sdf/placeable_refined",
                      *optimized_place_box, {255, 0, 0, 200}, 0.01f,
                      rerun::components::FillMode::Solid);

        } else {
          std::cout << "No valid placement pose found!!!!!" << std::endl;
        }
      }

    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
    }
  }

  void callbackMarkerDetection(
      const sensor_msgs::msg::Image::ConstSharedPtr& image1_msg,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg) {
    auto detector = dklib::perception::detection::d3::createMarkerDetector<
        dklib::perception::detection::d3::ArucoMarkerTarget3DPoseDetector::
            Param>(dklib::perception::detection::d3::MarkerTargetType::ARUCO,
                   {cv::aruco::DICT_4X4_50});
    cv::Mat rgb_img = cv_bridge::toCvShare(image1_msg, "bgr8")->image;
    Eigen::Matrix3d camera_intrinsics =
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
            camera_info_msg->k.data());

    auto result = detector->detectMarkerWithID(
        rgb_img, 47, 0.04, camera_intrinsics, {0, 0, 0, 0, 0});
    if (result) {
      std::cout << "Detected marker pose: \n" << result->matrix() << std::endl;
    } else {
      std::cout << "Marker not detected." << std::endl;
    }
  }

  void callbackPublishMarkerDetectionRerun() {
    Eigen::Matrix4d marker_pose = Eigen::Matrix4d::Identity();
    marker_pose << -0.113459385327, -0.988681482824, -0.0981625865706,
        -0.111166894074, -0.954312105941, 0.135935454724, -0.266101402859,
        0.0366779660149, 0.276433305403, 0.0634860431118, -0.958933861116,
        0.997429574978, 0, 0, 0, 1;
    Eigen::Isometry3d marker_transform(marker_pose);
    Eigen::Isometry3d inv_marker_transform = marker_transform.inverse();
    publishData(rec_, "/marker_47/camera", inv_marker_transform);
  }

  void callback(
      const sensor_msgs::msg::Image::ConstSharedPtr& image1_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr& image2_msg,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg) {
    // callbackMarkerDetection(image1_msg, camera_info_msg);
    callbackPublishMarkerDetectionRerun();
    callbackBoxDetection(image1_msg, image2_msg, camera_info_msg);
  }

 private:
  const float depth_min_threshold_ = 0.2;
  const float depth_max_threshold_ = 5.0;
  const float depth_factor_ = 1000.0;

  size_t queue_size_ = 10;
  bool use_exact_time_ = false;
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

  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr box_marker_pub_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;

  // Synchronization policies
  std::shared_ptr<ApproxSyncPolicy> approx_policy_;
  std::shared_ptr<ExactSyncPolicy> exact_policy_;
  std::shared_ptr<ApproxSync> approx_sync_;
  std::shared_ptr<ExactSync> exact_sync_;

  std::shared_ptr<rerun::RecordingStream> rec_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;

  auto node = std::make_shared<RGBDProcessNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
