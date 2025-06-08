#include <pcl/PolygonMesh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/vtk_smoothing/vtk_mesh_smoothing_laplacian.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/impl/point_types.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>

class DepthMeshingNode : public rclcpp::Node {
 public:
  DepthMeshingNode() : Node("depth_meshing_node") {
    // Initialize the node, set up publishers, subscribers, etc.
    RCLCPP_INFO(this->get_logger(), "Depth Meshing Node has been started.");

    point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "input_point_cloud", 10,
        std::bind(&DepthMeshingNode::processPointCloud, this,
                  std::placeholders::_1));

    // Publisher for mesh visualization
    mesh_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "mesh_visualization", 10);

    // Declare parameters for Poisson reconstruction
    this->declare_parameter("poisson_depth", 9);
    this->declare_parameter("normal_estimation_search_radius", 0.05);
    this->declare_parameter("output_mesh_path", "/tmp/reconstructed_mesh.obj");
  }

  ~DepthMeshingNode() {
    RCLCPP_INFO(this->get_logger(), "Depth Meshing Node is shutting down.");
  }

  void processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Convert PointCloud2 message to PCL format
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*msg, *cloud);

    // Process the point cloud (e.g., filtering, meshing)
    RCLCPP_INFO(this->get_logger(), "Received point cloud with %zu points.",
                cloud->size());

    // Remove any NaN points
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
    RCLCPP_INFO(this->get_logger(), "After removing NaN points: %zu points.",
                cloud->size());

    if (cloud->empty()) {
      RCLCPP_WARN(
          this->get_logger(),
          "Point cloud is empty after filtering, skipping mesh generation");
      return;
    }

    // Mesh the point cloud
    generateMesh(cloud, msg->header);
  }

  void generateMesh(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                    const std_msgs::msg::Header& header) {
    // // Step 1: Estimate normals for the point cloud
    RCLCPP_INFO(this->get_logger(), "Estimating normals...");
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(
        new pcl::PointCloud<pcl::PointNormal>);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    n.setInputCloud(cloud);
    n.setSearchMethod(tree);
    n.setKSearch(20);
    n.compute(*normals);

    RCLCPP_INFO(this->get_logger(), "Computed %zu normals.", normals->size());

    // Combine points and normals
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    // Step 2: Perform Poisson surface reconstruction
    RCLCPP_INFO(this->get_logger(),
                "Performing Poisson surface reconstruction...");

    // Create search tree
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(
        new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud(cloud_with_normals);

    // pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    // gp3.setSearchRadius(0.025);  // Set search radius for triangulation
    // gp3.setMu(2.5);
    // gp3.setMaximumNearestNeighbors(100);
    // gp3.setMaximumSurfaceAngle(M_PI / 4);  // 45 degrees
    // gp3.setMinimumAngle(M_PI / 18);        // 10 degrees
    // gp3.setMaximumAngle(2 * M_PI / 3);     // 120 degrees
    // gp3.setNormalConsistency(false);

    // gp3.setInputCloud(cloud_with_normals);
    // gp3.setSearchMethod(tree2);

    pcl::Poisson<pcl::PointNormal> poisson;

    // // Get depth parameter for Poisson reconstruction
    int poisson_depth = this->get_parameter("poisson_depth").as_int();
    poisson.setDepth(poisson_depth);
    poisson.setInputCloud(cloud_with_normals);

    pcl::PolygonMesh mesh;
    poisson.reconstruct(mesh);

    // RCLCPP_INFO(this->get_logger(),
    //             "Mesh reconstruction completed. Polygons: %zu",
    //             mesh.polygons.size());

    // // Save mesh to file if requested
    // std::string output_path =
    //     this->get_parameter("output_mesh_path").as_string();
    // if (!output_path.empty()) {
    //   RCLCPP_INFO(this->get_logger(), "Saving mesh to %s",
    //   output_path.c_str()); pcl::io::saveOBJFile(output_path, mesh);
    // }

    // // Publish mesh as marker for visualization
    publishMeshMarker(mesh, header);
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
    RCLCPP_INFO(this->get_logger(),
                "Mesh visualization published with %zu points.",
                marker.points.size());
  }

  // Add methods for processing point clouds, meshing, etc.
 private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      point_cloud_sub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mesh_publisher_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DepthMeshingNode>());
  rclcpp::shutdown();
  return 0;
}