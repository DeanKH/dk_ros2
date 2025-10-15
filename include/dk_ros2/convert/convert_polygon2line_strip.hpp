#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <dk_perception/detection/d3/radial_extremum_detector.hpp>
#include <vector>
#include <visualization_msgs/msg/marker.hpp>

template <typename PointT>
visualization_msgs::msg::Marker convertPolygon2LineStripMarker(
    const typename pcl::PointCloud<PointT>::Ptr& polygon) {
  visualization_msgs::msg::Marker marker;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.scale.x = 0.01;  // Line width
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 1.0;                        // Fully opaque
  marker.pose.orientation.w = 1.0;             // No rotation
  marker.points.reserve(polygon->size() + 1);  // +1 to close the loop
  for (size_t i = 0; i <= polygon->size(); ++i) {
    const auto& pt = polygon->points[i % polygon->size()];
    geometry_msgs::msg::Point p;
    p.x = pt.x;
    p.y = pt.y;
    p.z = pt.z;
    marker.points.push_back(p);
  }
  return marker;
}

visualization_msgs::msg::Marker convertBoundingBox3DMarker(
    const dklib::perception::detection::d3::BoundingBox3D& bbox) {
  visualization_msgs::msg::Marker marker;
  marker.type = visualization_msgs::msg::Marker::CUBE;
  marker.scale.x = bbox.size.x();
  marker.scale.y = bbox.size.y();
  marker.scale.z = bbox.size.z();
  marker.color.r = 0.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;
  marker.color.a = 0.5;  // Semi-transparent
  marker.pose.position.x = bbox.center.x();
  marker.pose.position.y = bbox.center.y();
  marker.pose.position.z = bbox.center.z();
  marker.pose.orientation.x = bbox.orientation.x();
  marker.pose.orientation.y = bbox.orientation.y();
  marker.pose.orientation.z = bbox.orientation.z();
  marker.pose.orientation.w = bbox.orientation.w();
  return marker;
}