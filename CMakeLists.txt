cmake_minimum_required(VERSION 3.10)
project(Pose6D)

# 17 is required by the ROS2 (rclcpp/tf2_ros) headers pulled in by capture.cpp;
# it is a strict superset of 11 for every other translation unit here.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THIRDPARTY_DIR "${CMAKE_SOURCE_DIR}/3rdparty")
message(STATUS "THIRDPARTY_DIR: ${THIRDPARTY_DIR}")

set(OpenCV_DIR "${THIRDPARTY_DIR}/opencv/lib/cmake/opencv4")
if (NOT EXISTS ${OpenCV_DIR})
    message(WARNING "OpenCV path does not exist: ${OpenCV_DIR}. Trying system OpenCV...")
    find_package(OpenCV REQUIRED)
else()
    find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
endif()

# 查找 Eigen
set(EIGEN3_DIR "${THIRDPARTY_DIR}/eigen/share/eigen3/cmake")
find_path(EIGEN3_INCLUDE_DIR Eigen/Core HINTS "${THIRDPARTY_DIR}/eigen/include/eigen3")
message(STATUS "EIGEN3_DIR: ${EIGEN3_DIR}")
message(STATUS "EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")
if (NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen 3.4.0 not found. Please install Eigen in ${THIRDPARTY_DIR}/eigen.")
endif()

find_library(REALSENSE2_LIBRARY NAMES realsense2 HINTS "${THIRDPARTY_DIR}/camera_realsense/lib")
find_path(REALSENSE2_INCLUDE_DIR librealsense2/rs.hpp HINTS "${THIRDPARTY_DIR}/camera_realsense/include")
message(STATUS "REALSENSE2_LIBRARY: ${REALSENSE2_LIBRARY}")
message(STATUS "REALSENSE2_INCLUDE_DIR: ${REALSENSE2_INCLUDE_DIR}")
if (NOT REALSENSE2_LIBRARY OR NOT REALSENSE2_INCLUDE_DIR)
    message(FATAL_ERROR "librealsense2 not found. Please install librealsense2 v2.54.2 in ${THIRDPARTY_DIR}/camera_realsense.")
endif()

# capture.cpp (RealSense + ROS2 TF data capture tool, meant to run on the
# robot's onboard computer) is the only thing that needs ROS2. calib/detect
# are pure OpenCV/Eigen offline tools and must keep building fine on a
# machine with no ROS2 install at all (e.g. your local calibration-solving
# machine). So: look for ROS2, but don't require it -- just skip the
# `capture` target if it's not there. Source ROS2 before running cmake
# (e.g. `source /opt/ros/humble/setup.bash`) to have it picked up.
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(tf2 QUIET)
find_package(tf2_ros QUIET)
find_package(geometry_msgs QUIET)
set(HAND_EYE_BUILD_CAPTURE FALSE)
if (ament_cmake_FOUND AND rclcpp_FOUND AND tf2_FOUND AND tf2_ros_FOUND AND geometry_msgs_FOUND)
    set(HAND_EYE_BUILD_CAPTURE TRUE)
else()
    message(STATUS "ROS2 (rclcpp/tf2_ros/geometry_msgs) not found -- skipping the "
                    "`capture` target. This is expected on a machine that only "
                    "runs `calib`/`detect`; source your ROS2 install if you did "
                    "mean to build `capture` here.")
endif()

include_directories(
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${REALSENSE2_INCLUDE_DIR}
)

add_library(common STATIC
        src/common/Math.cpp
        src/common/CoordinateTransformer.cpp
        src/common/ArucoMarkerDetector.cpp
        src/calibration/01_intrinsic/IntrinsicCalibrator.cpp
        src/calibration/01_intrinsic/ChessBoardCalibrator.cpp
        src/calibration/02_hand_eye_2d_4point/FourPointCalibrator.cpp
        src/calibration/03_hand_eye_2d_12point/TwelvePointCalibrator.cpp
        src/calibration/04_hand_eye_3d_ball/BallCalibrator.cpp
        src/calibration/05_hand_eye_3d_board_img/BoardImageCalibrator.cpp
        src/calibration/06_hand_eye_3d_board_cloud/BoardCloudCalibrator.cpp
        src/calibration/07_hand_eye_3d_multi_point/MultiPoint3DCalibrator.cpp
        src/calibration/hand_eye_3d_base/RobotCameraCalibrator3D.cpp
)

add_executable(calib calib.cpp)
target_link_libraries(calib common ${OpenCV_LIBS} ${REALSENSE2_LIBRARY})

add_executable(detect detect.cpp)
target_link_libraries(detect common ${OpenCV_LIBS} ${REALSENSE2_LIBRARY})

if (HAND_EYE_BUILD_CAPTURE)
    add_executable(capture capture.cpp)
    target_link_libraries(capture common ${OpenCV_LIBS} ${REALSENSE2_LIBRARY})
    ament_target_dependencies(capture rclcpp tf2 tf2_ros geometry_msgs)
endif()
