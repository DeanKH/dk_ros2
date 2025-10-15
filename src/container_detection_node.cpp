#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/ModelCoefficients.h>
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

#include <Eigen/Core>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/rclcpp.hpp>
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
  void callback(
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

      dklib::perception::type::pointcloud::
          IteratableColorizedPointCloudReadOnlyAccessor<
              dklib::perception::type::pointcloud::DepthImageSet>
              accessor(rgbd);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>());
      cloud->reserve(accessor.size());
      for (size_t i = 0; i < accessor.size(); ++i) {
        pcl::PointXYZRGB point;
        point.x = accessor.point_at(i)[0];
        point.y = accessor.point_at(i)[1];
        point.z = accessor.point_at(i)[2];
        auto color = accessor.color_at(i).value();
        point.r = color[2];
        point.g = color[1];
        point.b = color[0];
        cloud->push_back(point);
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
      float voxel_size = 0.005f;  // 20mm voxel size
      voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
      voxel_filter.filter(*cloud);

      // Remove outliers
      pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
      sor.setInputCloud(cloud);
      sor.setMeanK(50);
      sor.setStddevMulThresh(1.0);
      sor.filter(*cloud);

      // // crop box filter
      // pcl::CropBox<pcl::PointXYZRGB> box_filter;
      // box_filter.setInputCloud(cloud);

      // // Get parameters for crop box filter
      // double min_x = this->get_parameter("crop_box.min_x").as_double();
      // double min_y = this->get_parameter("crop_box.min_y").as_double();
      // double min_z = this->get_parameter("crop_box.min_z").as_double();
      // double max_x = this->get_parameter("crop_box.max_x").as_double();
      // double max_y = this->get_parameter("crop_box.max_y").as_double();
      // double max_z = this->get_parameter("crop_box.max_z").as_double();

      // Eigen::Vector4f min_point(min_x, min_y, min_z, 1.0);
      // Eigen::Vector4f max_point(max_x, max_y, max_z, 1.0);
      // box_filter.setMin(min_point);
      // box_filter.setMax(max_point);
      // box_filter.filter(*cloud);

      std::cout << "Point cloud size after outlier removal: " << cloud->size()
                << std::endl;

      const Eigen::Vector3f origin = [&cloud]() {
        return Eigen::Vector3f::Zero();
        // pcl::PointXYZ origin_pt;
        // pcl::computeCentroid<pcl::PointXYZRGB, pcl::PointXYZ>(*cloud,
        //                                                       origin_pt);
        // return Eigen::Vector3f(origin_pt.x, origin_pt.y, origin_pt.z);
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

      std::cout << "Detected " << min_points->size() << " minimum points."
                << std::endl;

      auto inline_points = detector.getInlinePoints();
      // publish point cloud
      sensor_msgs::msg::PointCloud2 output_msg;
      pcl::toROSMsg(*inline_points, output_msg);
      output_msg.header = image1_msg->header;
      pointcloud_pub_->publish(output_msg);

      visualization_msgs::msg::Marker marker =
          convertPolygon2LineStripMarker<pcl::PointXYZRGB>(min_points);
      marker.header = image1_msg->header;
      marker.ns = "container_area";
      marker.id = 0;
      marker_pub_->publish(marker);

      visualization_msgs::msg::Marker box_marker =
          convertBoundingBox3DMarker(bbox);
      box_marker.header = image1_msg->header;
      box_marker.ns = "container_box";
      box_marker.id = 0;
      box_marker_pub_->publish(box_marker);
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
    }
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
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;

  auto node = std::make_shared<RGBDProcessNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
