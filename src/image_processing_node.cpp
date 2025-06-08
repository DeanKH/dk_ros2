#include <cstdio>
#include <dk_perception/dnn/superpoint.hpp>
#include <memory>
#include <rclcpp/node.hpp>
#include <rclcpp/rclcpp.hpp>

#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/msg/image.hpp"

class ImageProcessingNode : public rclcpp::Node {
 public:
  ImageProcessingNode(const std::string &node_name = "image_processing_node")
      : Node(node_name) {
    // Create a subscriber with image topic name "image_raw"
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "image_raw", 10,
        std::bind(&ImageProcessingNode::image_callback, this,
                  std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Image Processing Node has been started");
  }

 private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) const {
    RCLCPP_INFO(this->get_logger(), "Received an image with size: %dx%d",
                msg->width, msg->height);

    // Here you can add your image processing code
    // For example, converting ROS Image message to OpenCV image:
    // cv_bridge::CvImagePtr cv_ptr;
    // try {
    //   cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
    // } catch (cv_bridge::Exception& e) {
    //   RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    //   return;
    // }

    // Process the image with OpenCV
    // cv::Mat processed_image = cv_ptr->image;
    // ...
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImageProcessingNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}