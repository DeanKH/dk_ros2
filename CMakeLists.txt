cmake_minimum_required(VERSION 3.8)
project(dk_ros2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(dk_perception REQUIRED)

find_package(onnxruntime REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(message_filters REQUIRED)
find_package(ament_index_cpp REQUIRED)

# add_executable(depth_meshing_node src/depth_meshing_node.cpp)
# target_include_directories( depth_meshing_node PUBLIC
# $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
# $<INSTALL_INTERFACE:include> ${Eigen3_INCLUDE_DIRS})
# ament_target_dependencies( depth_meshing_node "rclcpp" "sensor_msgs"
# "geometry_msgs" "visualization_msgs" "dk_perception" "PCL" "cv_bridge"
# "pcl_conversions" "ament_index_cpp") target_link_libraries(depth_meshing_node
# ${PCL_LIBRARIES} ${OpenCV_LIBS})

# add_executable(image_processing_node src/image_processing_node.cpp)
# target_include_directories( image_processing_node PUBLIC
# $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
# $<INSTALL_INTERFACE:include> ${Eigen3_INCLUDE_DIRS})

# ament_target_dependencies( image_processing_node "rclcpp" "sensor_msgs"
# "geometry_msgs" "visualization_msgs" "dk_perception" "PCL" "cv_bridge")
# target_link_libraries(image_processing_node dk_perception::dnn
# onnxruntime::onnxruntime ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(rectangle3d_detection_node src/rectangle3d_detection_node.cpp)
target_include_directories(
  rectangle3d_detection_node
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include> ${Eigen3_INCLUDE_DIRS})
ament_target_dependencies(
  rectangle3d_detection_node
  "rclcpp"
  "sensor_msgs"
  "geometry_msgs"
  "visualization_msgs"
  "dk_perception"
  "PCL"
  "cv_bridge"
  "pcl_conversions"
  "message_filters")
target_link_libraries(
  rectangle3d_detection_node ${PCL_LIBRARIES} ${OpenCV_LIBS}
  dk_perception::type dk_perception::dnn dk_perception::detection)

# image_processing_node depth_meshing_node
install(TARGETS rectangle3d_detection_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights comment the
  # line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo) comment the line
  # when this package is in a git repo and when a copyright and license is added
  # to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
