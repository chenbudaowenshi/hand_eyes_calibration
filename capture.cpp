// Hand-eye calibration data capture tool.
//
// Grabs a color frame from a RealSense camera via the SDK directly (so the
// saved intrinsics/depth-scale/resolution/fps are what the device actually
// reports, not a config file's claim about it) and, on operator trigger,
// pairs that frame with the robot's head_pitch_Link pose read from ROS2 TF
// at that instant. Output layout matches what calib.cpp expects:
//
//   <dataset>/rgb/frame_%04d.png
//   <dataset>/poses.csv               (image_name,capture_stamp_s,x_mm,y_mm,z_mm,rx_deg,ry_deg,rz_deg)
//   <dataset>/realsense_intrinsics.xml (K, distortion, depth_scale, width, height, fps)
//
// poses.csv rows are T_<base-frame>_<gripper-frame> (default
// T_waist_yaw_Link_head_pitch_Link; translation in mm, RPY in degrees,
// extrinsic/fixed-axis X-Y-Z -- i.e. tf2::Matrix3x3::getRPY()'s own
// convention, matching what calib.cpp's RobotCameraCalibrator3D expects).
// Whatever --base-frame is chosen must stay physically fixed relative to
// the calibration board for the whole capture session.
#include "common/RealSenseCamera.h"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/time.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <utility>

namespace {

constexpr double kMm = 1000.0;
constexpr double kRadToDeg = 180.0 / M_PI;
const cv::Size kBoardCorners(11, 8); // 12x9 squares -> 11x8 inner corners
constexpr double kSquareSizeMm = 15.0;

bool detect_board(const cv::Mat &image, std::vector<cv::Point2f> *corners = nullptr) {
  if (image.empty())
    return false;
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  std::vector<cv::Point2f> found;
  bool ok = cv::findChessboardCorners(
      gray, kBoardCorners, found,
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
  if (ok) {
    cv::cornerSubPix(
        gray, found, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                         30, 0.001));
  }
  if (corners != nullptr)
    *corners = std::move(found);
  return ok;
}

int count_existing_frames(const std::string &dataset_dir) {
  std::vector<std::string> matches;
  cv::glob(dataset_dir + "/rgb/frame_*.png", matches, false);
  return static_cast<int>(matches.size());
}

void write_realsense_intrinsics(const std::string &path,
                                const RealSenseCameraInfo &info) {
  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  fs << "K" << info.K << "distortion" << info.dist << "depth_scale"
     << info.depth_scale << "width" << info.width << "height" << info.height
     << "fps" << info.fps << "note"
     << "K/distortion/depth_scale/width/height/fps as reported live by the "
        "RealSense SDK at capture time.";
  fs.release();
  std::cout << "[INFO] RealSense intrinsics saved to: " << path << "\n"
            << "  K = " << info.K << "\n  distortion = " << info.dist << "\n"
            << "  depth_scale = " << info.depth_scale
            << " m/LSB, resolution = " << info.width << "x" << info.height
            << " @ " << info.fps << " fps" << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  std::string dataset_dir = "../dataset";
  // waist_yaw_Link, not base_link: per this robot's URDF, head_pitch_Link
  // hangs off base_link through 6 moving joints (2 lift-column prismatic +
  // waist_pitch + waist_yaw + head_yaw + head_pitch), all of which would
  // need live, accurate joint states for TF to be trustworthy. Only
  // head_yaw_joint and head_pitch_joint sit between waist_yaw_Link and
  // head_pitch_Link -- exactly the two joints you're already sweeping for
  // rotational diversity -- so this needs far less of the robot's control
  // stack to be live. Valid as long as the lift column and waist are
  // physically held still for the whole capture session.
  std::string base_frame = "waist_yaw_Link";
  std::string gripper_frame = "head_pitch_Link";
  int width = 1920, height = 1080, fps = 30;
  // No DISPLAY (typical for a plain `ssh onboard-pc` session, no -X/-Y) ->
  // default to headless so we never touch GTK/X11 at all. Either flag below
  // can still force the choice explicitly.
  bool headless = std::getenv("DISPLAY") == nullptr;

  for (int i = 1; i < argc; ++i) {
    std::string argument(argv[i]);
    if (argument == "--dataset" && i + 1 < argc)
      dataset_dir = argv[++i];
    else if (argument == "--base-frame" && i + 1 < argc)
      base_frame = argv[++i];
    else if (argument == "--gripper-frame" && i + 1 < argc)
      gripper_frame = argv[++i];
    else if (argument == "--width" && i + 1 < argc)
      width = std::atoi(argv[++i]);
    else if (argument == "--height" && i + 1 < argc)
      height = std::atoi(argv[++i]);
    else if (argument == "--fps" && i + 1 < argc)
      fps = std::atoi(argv[++i]);
    else if (argument == "--headless")
      headless = true;
    else if (argument == "--gui")
      headless = false;
    else if (argument == "--help") {
      std::cout
          << "Usage: capture --dataset DIR --base-frame FRAME_ID "
             "[--gripper-frame FRAME_ID] [--width W] [--height H] [--fps F] "
             "[--headless|--gui]\n"
          << "GUI mode: live-previews the RealSense color stream, SPACE to "
             "capture\n"
          << "a (image, head_pitch_Link pose) sample, 'q'/ESC to quit.\n"
          << "Headless mode (auto-selected when $DISPLAY is unset, e.g. a "
             "plain\n"
          << "`ssh onboard-pc` session): no preview window; press Enter to "
             "capture,\n"
          << "type 'q' + Enter to quit. Use this over SSH without X "
             "forwarding.\n"
          << "--base-frame must name a TF frame that stays fixed relative\n"
          << "to the calibration board for the whole session. Default "
             "waist_yaw_Link\n"
          << "only needs head_yaw_joint/head_pitch_joint to be live -- hold "
             "the\n"
          << "lift column and waist still and only move the head. Use "
             "base_link\n"
          << "instead only once the whole-body joint states are reliably "
             "published.\n";
      return 0;
    } else {
      std::cerr << "Unknown or incomplete argument: " << argument << "\n";
      return 2;
    }
  }

  if (cv::utils::fs::createDirectories(dataset_dir + "/rgb") != true) {
    std::cerr << "Cannot create dataset directory: " << dataset_dir << "\n";
    return 2;
  }

  rclcpp::init(0, nullptr);
  auto node = std::make_shared<rclcpp::Node>("hand_eye_capture");
  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  // spin_thread=true: the listener runs its own executor thread, so /tf and
  // /tf_static keep flowing in without us having to spin `node` ourselves.
  auto tf_listener =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer, node, true);

  RealSenseCamera camera;
  if (!camera.start(width, height, fps)) {
    std::cerr << "[ERROR] Could not start the RealSense pipeline at " << width
              << "x" << height << "@" << fps << "fps. Check that:\n"
              << "  - the camera is physically plugged in and shows up in "
                 "`lsusb`/`ls /dev/video*` (see the rs2::error above -- "
                 "\"Couldn't resolve requests\" can also mean the *color* "
                 "resolution/fps you asked for isn't one this specific "
                 "model supports, even though the depth stream is now "
                 "requested at a fixed, broadly-supported 640x480),\n"
              << "  - no other process already has it open (stop any other "
                 "camera driver node first), and\n"
              << "  - try a resolution/fps your model definitely supports, "
                 "e.g. --width 640 --height 480 --fps 30.\n";
    rclcpp::shutdown();
    return 1;
  }

  std::cout << "[INFO] Waiting for the first color frame...\n";
  constexpr auto kFirstFrameTimeout = std::chrono::seconds(10);
  auto wait_start = std::chrono::steady_clock::now();
  while (rclcpp::ok() && camera.getLatestColor().empty()) {
    if (std::chrono::steady_clock::now() - wait_start > kFirstFrameTimeout) {
      std::cerr << "[ERROR] No color frame received within "
                << kFirstFrameTimeout.count()
                << "s of a successful pipeline start -- the device likely "
                   "stalled (USB bandwidth/cable issue). Aborting instead of "
                   "hanging forever.\n";
      camera.stop();
      rclcpp::shutdown();
      return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (!rclcpp::ok()) {
    camera.stop();
    return 1;
  }

  write_realsense_intrinsics(dataset_dir + "/realsense_intrinsics.xml",
                             camera.getCameraInfo());

  const std::string csv_path = dataset_dir + "/poses.csv";
  bool csv_exists = std::ifstream(csv_path).good();
  std::ofstream csv(csv_path, std::ios::app);
  if (!csv.is_open()) {
    std::cerr << "Cannot open poses.csv for append: " << csv_path << "\n";
    return 2;
  }
  if (!csv_exists) {
    csv << "# image_name,capture_stamp_s,x_mm,y_mm,z_mm,rx_deg,ry_deg,rz_deg "
          "-- T_"
       << base_frame << "_" << gripper_frame
       << ", RPY extrinsic X-Y-Z (deg)\n";
  }

  int index = count_existing_frames(dataset_dir);
  std::cout << "[INFO] Resuming at frame index " << index << ".\n"
            << "[INFO] Looking up TF: " << base_frame << " -> " << gripper_frame
            << " (must stay fixed w.r.t. the calibration board!)\n"
            << "[INFO] Mode: " << (headless ? "headless (no preview window)"
                                           : "GUI preview")
            << "\n";

  // Grabs whatever color frame + TF is current *right now* and appends one
  // row; shared by both the GUI and headless interaction loops below so
  // "press SPACE" and "press Enter" behave identically.
  auto try_capture_sample = [&]() -> bool {
    cv::Mat color;
    double stamp_s = 0.0;
    if (!camera.getLatestColorWithTimestamp(color, stamp_s)) {
      std::cerr << "[WARN] No color frame available yet, sample discarded.\n";
      return false;
    }
    if (!detect_board(color)) {
      std::cerr << "[WARN] 12x9 chessboard (11x8 inner corners) not detected; "
                   "sample discarded.\n";
      return false;
    }

    geometry_msgs::msg::TransformStamped transform;
    try {
      transform = tf_buffer->lookupTransform(base_frame, gripper_frame,
                                             rclcpp::Time(static_cast<int64_t>(
                                                 stamp_s * 1e9)),
                                             tf2::durationFromSec(0.5));
    } catch (const std::exception &e) {
      std::cerr << "[WARN] TF lookup " << base_frame << " -> " << gripper_frame
                << " failed: " << e.what() << " -- sample discarded.\n";
      return false;
    }

    const auto &t = transform.transform.translation;
    tf2::Quaternion q(transform.transform.rotation.x,
                      transform.transform.rotation.y,
                      transform.transform.rotation.z,
                      transform.transform.rotation.w);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    char name[64];
    std::snprintf(name, sizeof(name), "frame_%04d.png", index);
    if (!cv::imwrite(dataset_dir + "/rgb/" + name, color)) {
      std::cerr << "[WARN] Failed to write image, sample discarded.\n";
      return false;
    }

    csv << name << "," << std::setprecision(17) << stamp_s << ","
        << (t.x * kMm) << "," << (t.y * kMm) << "," << (t.z * kMm) << ","
        << (roll * kRadToDeg) << "," << (pitch * kRadToDeg) << ","
        << (yaw * kRadToDeg) << "\n";
    csv.flush();

    std::cout << "[CAPTURED] #" << index << " -> " << name << "  pose(mm,deg)=["
              << (t.x * kMm) << ", " << (t.y * kMm) << ", " << (t.z * kMm)
              << ", " << (roll * kRadToDeg) << ", " << (pitch * kRadToDeg)
              << ", " << (yaw * kRadToDeg) << "]\n";
    ++index;
    return true;
  };

  if (headless) {
    std::cout << "[INFO] Enter = capture, 'q' + Enter = quit.\n";
    std::string line;
    while (rclcpp::ok() && std::getline(std::cin, line)) {
      if (line == "q" || line == "quit")
        break;
      try_capture_sample();
    }
  } else {
    const std::string window = "capture (SPACE=save, q=quit)";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    std::cout << "[INFO] SPACE = capture, q/ESC = quit.\n";
    while (rclcpp::ok()) {
      cv::Mat color = camera.getLatestColor();
      if (color.empty()) {
        cv::waitKey(10);
        continue;
      }
      std::vector<cv::Point2f> corners;
      cv::Mat preview = color.clone();
      bool board_found = detect_board(color, &corners);
      cv::drawChessboardCorners(preview, kBoardCorners, corners, board_found);
      cv::putText(preview,
                  board_found ? "BOARD OK - SPACE saves" : "SHOW 12x9 BOARD",
                  cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                  board_found ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 255),
                  2, cv::LINE_AA);
      cv::imshow(window, preview);
      int key = cv::waitKey(15) & 0xFF;
      if (key == 'q' || key == 27)
        break;
      if (key != ' ' && key != 's')
        continue;
      try_capture_sample();
    }
  }

  camera.stop();
  rclcpp::shutdown();
  std::cout << "[INFO] Capture session complete: " << index << " samples in "
            << dataset_dir << "\n";
  return 0;
}
