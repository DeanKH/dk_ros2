#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/PolygonMesh.h>
#include <pcl/surface/poisson.h>

#include <Eigen/Core>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <dk_perception/detection/d3/rectangle_detection.hpp>
#include <dk_perception/dnn/superpoint.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "dk_perception/detection/d2/node_extractor.hpp"
#include "dk_perception/type/pointcloud/rgbd_type.hpp"

template <typename ImageType>
void minmaxFilter(cv::Mat& img, ImageType min, ImageType max) {
  cv::Mat mask = (img >= min) & (img <= max);
  img.setTo(0, ~mask);  // Set pixels outside the range to 0
}

class Rectangle3dDetectionNode : public rclcpp::Node {
 public:
  explicit Rectangle3dDetectionNode(const rclcpp::NodeOptions& options)
      : Node("rectangle3d_detection_node", options) {
    RCLCPP_INFO(this->get_logger(), "Starting Rectangle3D Detection Node");

    // Declare parameters
    this->declare_parameter("use_exact_time", false);
    this->declare_parameter("queue_size", 10);
    this->declare_parameter("depth_factor", 1000.0);
    this->declare_parameter("depth_min_threshold", 0.75);
    this->declare_parameter("depth_max_threshold", 2.0);

    this->declare_parameter("superpoint_vocab_dict_path",
                            "focus_descriptors.csv");
    this->declare_parameter("superpoint_model_path", "superpoint.onnx");
    this->declare_parameter("superpoint_vocab_similarity_threshold", 0.4);
    this->declare_parameter("superpoint_min_score", 0.0005);
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

      RCLCPP_INFO(this->get_logger(),
                  "Using Exact Time synchronization policy");
    } else {
      // Use approximate time synchronization (default)
      approx_policy_ = std::make_shared<ApproxSyncPolicy>(queue_size);
      approx_sync_ = std::make_shared<ApproxSync>(*approx_policy_);
      approx_sync_->connectInput(image1_sub_, image2_sub_, camera_info_sub_);
      approx_sync_->registerCallback(std::bind(
          &Rectangle3dDetectionNode::callback, this, std::placeholders::_1,
          std::placeholders::_2, std::placeholders::_3));

      RCLCPP_INFO(this->get_logger(),
                  "Using Approximate Time synchronization policy");
    }
  }

 private:
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
    RCLCPP_INFO(this->get_logger(), "Received synchronized messages");
    RCLCPP_INFO(this->get_logger(), "Image1 timestamp: %u.%u",
                image1_msg->header.stamp.sec, image1_msg->header.stamp.nanosec);
    RCLCPP_INFO(this->get_logger(), "Image2 timestamp: %u.%u",
                image2_msg->header.stamp.sec, image2_msg->header.stamp.nanosec);
    RCLCPP_INFO(this->get_logger(), "CameraInfo timestamp: %u.%u",
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
          static_cast<float>(depth_factor_)};

      // Process images and camera info
      processImages(rgbd);
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
    }
  }

  // Process the synchronized images and camera info
  void processImages(
      const dklib::perception::type::pointcloud::DepthImageSet& rgbd) {
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
    cv::Mat draw_image = rgbd.color_image().clone();
    for (const auto& point : points) {
      cv::circle(draw_image, point, 3, cv::Scalar(0, 255, 0), -1);
    }
    cv::imshow("Detected Keypoints", draw_image);
    cv::waitKey(1);

    pcl::Poisson<pcl::PointNormal> poisson_reconstructor_;
    pcl::PolygonMesh mesh;

    dklib::perception::detection::d3::RectangleDetection<
        pcl::Poisson<pcl::PointNormal>>
        detector;
    // Example: You would use your rectangle detection functionality here
    // dk_perception::detection::d3::RectangleDetection detector;
    // auto rectangles = detector.detect(image1, image2, camera_info);

    // Process detection results...
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
