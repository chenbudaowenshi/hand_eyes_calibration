#include "calibration/01_intrinsic/IntrinsicCalibrator.h"
#include "calibration/02_hand_eye_2d_4point/FourPointCalibrator.h"
#include "calibration/03_hand_eye_2d_12point/TwelvePointCalibrator.h"
#include "calibration/04_hand_eye_3d_ball/BallCalibrator.h"
#include "calibration/05_hand_eye_3d_board_img/BoardImageCalibrator.h"
#include "calibration/06_hand_eye_3d_board_cloud/BoardCloudCalibrator.h"
#include "calibration/07_hand_eye_3d_multi_point/MultiPoint3DCalibrator.h"
#include "calibration/hand_eye_3d_base/RobotCameraCalibrator3D.h"
#include "common/CoordinateTransformer.h"
#include <algorithm>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <opencv2/core/utils/filesystem.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

#define RUN_INTRINSIC 1
#define RUN_FOUR_POINT_2D 0
#define RUN_TWELVE_POINT_2D 0
#define RUN_HAND_EYE_3D_BALL 0
#define RUN_HAND_EYE_3D_BOARD_IMG 1
#define RUN_HAND_EYE_3D_BOARD_CLOUD 0
#define RUN_MULTI_POINT_3D 0

static std::vector<std::string> list_images(const std::string &directory) {
  std::vector<std::string> result;
  const char *extensions[] = {"*.png", "*.jpg", "*.jpeg", "*.bmp"};
  for (const char *extension : extensions) {
    std::vector<std::string> matches;
    cv::glob(directory + "/" + extension, matches, false);
    result.insert(result.end(), matches.begin(), matches.end());
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

// poses.csv format (one row per image, in the same sorted order as rgb/):
// image_name,capture_stamp_s,x_mm,y_mm,z_mm,rx_deg,ry_deg,rz_deg
// x/y/z are T_<base-frame>_head_pitch_Link translation (mm) -- base-frame
// is whatever `capture --base-frame ...` was pointed at (default
// waist_yaw_Link; must be physically fixed relative to the calibration
// board for the whole capture session); rx/ry/rz are its roll/pitch/yaw
// (deg, extrinsic X-Y-Z / ROS "fixed axis" convention), as produced by the
// `capture` tool from ROS2 TF.
struct RobotPoseRecord {
  std::string image_name;
  double stamp_s = 0.0;
  Pose3D pose;
};

static std::vector<RobotPoseRecord> load_robot_poses(const std::string &path) {
  std::ifstream input(path);
  if (!input.is_open())
    throw std::runtime_error("cannot open robot pose CSV: " + path);

  std::vector<RobotPoseRecord> records;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    std::string image_name;
    double stamp = 0.0;
    double x = 0.0, y = 0.0, z = 0.0;
    double rx = 0.0, ry = 0.0, rz = 0.0;
    if (!(row >> image_name >> stamp >> x >> y >> z >> rx >> ry >> rz))
      continue;  // allow a header row
    RobotPoseRecord record;
    record.image_name = image_name;
    record.stamp_s = stamp;
    record.pose = Pose3D(x, y, z, rx, ry, rz);
    records.push_back(record);
  }
  return records;
}

// Returns the filename component of a path, tolerant of '/' and '\\'.
static std::string basename_of(const std::string &path) {
  size_t pos = path.find_last_of("/\\");
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

struct ErrorStats {
  double rms = 0.0;
  double avg = 0.0;
  double max = 0.0;
  size_t count = 0;
};

// Standardized RMS/avg/max over the non-negative entries of a per-point
// error vector (negative entries mark "not detected" and are excluded).
static ErrorStats compute_error_stats(const std::vector<double> &errors) {
  ErrorStats stats;
  double sum = 0.0, sum_sq = 0.0;
  for (double e : errors) {
    if (e < 0)
      continue;
    sum += e;
    sum_sq += e * e;
    stats.max = std::max(stats.max, e);
    stats.count++;
  }
  if (stats.count > 0) {
    stats.avg = sum / stats.count;
    stats.rms = std::sqrt(sum_sq / stats.count);
  }
  return stats;
}

static double compute_intrinsic_rms(const std::vector<std::string> &images,
                                    const CalibConfig &cfg,
                                    const cv::Mat &K,
                                    const cv::Mat &dist) {
  double sum_squared = 0.0;
  size_t count = 0;
  for (const auto &path : images) {
    cv::Mat image = cv::imread(path);
    if (image.empty())
      continue;
    cv::Mat corners_vis;
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> object_points;
    if (IntrinsicCalibrator::detectCalibBoard(
            image, corners_vis, image_points, object_points, cfg, 1) != 1)
      continue;
    cv::Mat rvec, tvec;
    if (!cv::solvePnP(object_points, image_points, K, dist, rvec, tvec))
      continue;
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec, tvec, K, dist, projected);
    for (size_t i = 0; i < image_points.size() && i < projected.size(); ++i) {
      const cv::Point2f delta = image_points[i] - projected[i];
      sum_squared += static_cast<double>(delta.dot(delta));
      ++count;
    }
  }
  return count == 0 ? -1.0 : std::sqrt(sum_squared / count);
}

// Re-runs board detection + PnP on a held-out sample (not part of the
// hand-eye solve) and maps the resulting marker position through the
// candidate hand-eye matrix (T_head_pitch_Link_camera_optical) and that
// sample's own robot pose (T_<base-frame>_head_pitch_Link) into the base
// frame, where it is compared against the fixed board position the
// solver estimated from the training samples. This is an honest
// out-of-sample check: nothing about this image or pose influenced the
// solved transform. EIH-only, matching this pipeline's calibration type.
struct ValidationResult {
  bool board_found = false;
  Vector3D predicted_point_base_frame;
  double error_mm = -1.0;
};

static ValidationResult
validate_held_out_sample(const std::string &image_path, Pose3D robot_pose,
                         const CalibConfig &cfg, const cv::Mat &K_first,
                         const cv::Mat &dist_first, const cv::Mat &K_second,
                         const cv::Mat &dist_second, Pose3D hand_eye_pose,
                         const Vector3D &fixed_point_base_frame) {
  ValidationResult result;
  cv::Mat raw = cv::imread(image_path);
  if (raw.empty())
    return result;

  cv::Mat undistorted;
  cv::undistort(raw, undistorted, K_first, dist_first);

  cv::Mat corners_vis;
  std::vector<cv::Point2f> pixel_corners;
  std::vector<cv::Point3f> world_corners;
  int ret = IntrinsicCalibrator::detectCalibBoard(
      undistorted, corners_vis, pixel_corners, world_corners, cfg, 1);
  if (ret != 1)
    return result;

  cv::Mat rvec, tvec;
  cv::solvePnP(world_corners, pixel_corners, K_second, dist_second, rvec,
              tvec);
  Eigen::Vector3d p_camera_optical(tvec.at<double>(0), tvec.at<double>(1),
                                   tvec.at<double>(2));

  Eigen::Matrix4d T_head_pitch_camera = hand_eye_pose.toTransform3D().eigen();
  Eigen::Vector3d p_head_pitch =
      T_head_pitch_camera.block<3, 3>(0, 0) * p_camera_optical +
      T_head_pitch_camera.block<3, 1>(0, 3);

  Eigen::Matrix4d T_base_head_pitch = robot_pose.toTransform3D().eigen();
  Eigen::Vector3d p_base = T_base_head_pitch.block<3, 3>(0, 0) * p_head_pitch +
                          T_base_head_pitch.block<3, 1>(0, 3);

  result.board_found = true;
  result.predicted_point_base_frame = Vector3D(p_base);
  result.error_mm = (p_base - fixed_point_base_frame.eigen()).norm();
  return result;
}

static void write_urdf_snippet(const std::string &path,
                               const Pose3D &pose_mm,
                               const std::string &parent,
                               const std::string &child,
                               const std::string &child_axis_note =
                                   "child optical frame (OpenCV/REP-103 "
                                   "convention: X right, Y down, Z forward "
                                   "out of the lens)") {
  constexpr double kPi = 3.14159265358979323846;
  std::ofstream output(path);
  if (!output.is_open())
    throw std::runtime_error("cannot write URDF snippet: " + path);
  output << "<!-- T_" << parent << "_" << child << ": maps a point expressed\n"
         << "     in the " << child_axis_note << " into the\n"
         << "     parent link frame. rpy is extrinsic (fixed-axis) X-Y-Z,\n"
         << "     matching URDF/ROS <origin rpy=\"...\"/> semantics. -->\n"
         << "<joint name=\"" << child << "_joint\" type=\"fixed\">\n"
         << "  <parent link=\"" << parent << "\"/>\n"
         << "  <child link=\"" << child << "\"/>\n"
         << std::fixed << std::setprecision(9)
         << "  <origin xyz=\"" << pose_mm.x() / 1000.0 << " "
         << pose_mm.y() / 1000.0 << " " << pose_mm.z() / 1000.0
         << "\" rpy=\"" << pose_mm.rx() * kPi / 180.0 << " "
         << pose_mm.ry() * kPi / 180.0 << " "
         << pose_mm.rz() * kPi / 180.0 << "\"/>\n"
         << "</joint>\n";
}

struct Logger {
  static void info(const std::string &msg) {
    std::cout << "\033[1;36m[INFO] " << msg << "\033[0m" << std::endl;
  }

  static void success(const std::string &msg) {
    std::cout << "\033[1;32m[SUCCESS] " << msg << "\033[0m" << std::endl;
  }

  static void warn(const std::string &msg) {
    std::cout << "\033[1;33m[WARN] " << msg << "\033[0m" << std::endl;
  }

  static void error(const std::string &msg) {
    std::cerr << "\033[1;31m[ERROR] " << msg << "\033[0m" << std::endl;
  }

  static void section(const std::string &msg) {
    std::cout << "\n\033[1;35m"
              << "============================================================="
                 "========"
              << "\n   " << msg << "\n"
              << "============================================================="
                 "========"
              << "\033[0m" << std::endl;
  }
};

int main(int argc, char **argv) {
  std::string dataset_dir = "../dataset";
  std::string output_dir = "../calibration_output";
  std::string child_frame = "head_d435i_optical_frame";
  std::string fixed_intrinsics_path;
  int validate_count = 1;
  for (int i = 1; i < argc; ++i) {
    std::string argument(argv[i]);
    if (argument == "--dataset" && i + 1 < argc)
      dataset_dir = argv[++i];
    else if (argument == "--output" && i + 1 < argc)
      output_dir = argv[++i];
    else if (argument == "--child-frame" && i + 1 < argc)
      child_frame = argv[++i];
    else if (argument == "--validate-count" && i + 1 < argc)
      validate_count = std::atoi(argv[++i]);
    else if (argument == "--fixed-intrinsics" && i + 1 < argc)
      fixed_intrinsics_path = argv[++i];
    else if (argument == "--help") {
      std::cout
          << "Usage: calib --dataset DATASET_DIR --output OUTPUT_DIR "
             "[--validate-count N] [--child-frame NAME] "
             "[--fixed-intrinsics PATH]\n"
          << "DATASET_DIR/rgb contains chessboard images and\n"
          << "DATASET_DIR/poses.csv contains "
             "image,stamp,x_mm,y_mm,z_mm,rx_deg,ry_deg,rz_deg.\n"
          << "The last N samples (default 1) are held out of both intrinsic\n"
          << "refinement and the hand-eye solve, then used to independently\n"
          << "validate the resulting matrix.\n"
          << "--fixed-intrinsics PATH: use a realsense_intrinsics.xml-style\n"
          << "K/distortion instead of re-fitting from these images. Prefer\n"
          << "this whenever the dataset is camera-moves/board-fixed (the\n"
          << "board stays close to fronto-parallel across all frames) --\n"
          << "that geometry is poorly conditioned for cv::calibrateCamera\n"
          << "and can silently produce a garbage fit (asymmetric fx/fy,\n"
          << "huge distortion) even when its own reported RMS looks small.\n";
      return 0;
    } else {
      std::cerr << "Unknown or incomplete argument: " << argument << "\n";
      return 2;
    }
  }
  if (validate_count < 0) {
    std::cerr << "--validate-count must be >= 0\n";
    return 2;
  }
  if (cv::utils::fs::createDirectories(output_dir) != true) {
    std::cerr << "Cannot create output directory: " << output_dir << "\n";
    return 2;
  }
  std::cout << std::fixed << std::setprecision(4);
  Logger::section("Industrial Camera + Robot Hand-Eye Calibration Program");
  std::cout << "Included: 2.5D Hand-Eye | 2D Four-Point | 2D Twelve-Point High "
               "Precision Calibration\n"
            << std::endl;

#if RUN_INTRINSIC
  CalibConfig cfg;

  // Physical board: 12x9 squares (12 across, 9 down), 15 mm per square.
  // OpenCV receives the number of *inner* corners, hence 11x8.
  cfg.pattern = CHESSBOARD;
  cfg.calib_type = CalibrationType3D::EIH; // head camera: camera moves with
                                            // head_pitch_Link, board is
                                            // fixed relative to whatever
                                            // --base-frame `capture` used.
  cfg.cols = 11;
  cfg.rows = 8;
  cfg.interval_mm = 15;         // square size in millimetres
  cfg.marker_length_mm = 50.0f; // 标定板marker长度
  cfg.xml_intrinsic_1st = output_dir + "/calib_intrinsic_1st.xml";
  cfg.xml_intrinsic_2nd = output_dir + "/calib_intrinsic_2nd.xml";
  cfg.xml_extrinsic = output_dir + "/calib_extrinsic.xml";

  if (!fixed_intrinsics_path.empty()) {
    cv::FileStorage fs_fixed(fixed_intrinsics_path, cv::FileStorage::READ);
    if (!fs_fixed.isOpened()) {
      std::cerr << "Cannot open --fixed-intrinsics file: "
                << fixed_intrinsics_path << "\n";
      return 2;
    }
    fs_fixed["K"] >> cfg.fixed_K;
    fs_fixed["distortion"] >> cfg.fixed_dist;
    fs_fixed.release();
    if (cfg.fixed_K.empty()) {
      std::cerr << "--fixed-intrinsics file has no K matrix: "
                << fixed_intrinsics_path << "\n";
      return 2;
    }
    cfg.use_fixed_intrinsics = true;
    Logger::info("Using fixed intrinsics from " + fixed_intrinsics_path +
                " (skipping cv::calibrateCamera refits).");
  }

  std::vector<std::string> all_images = list_images(dataset_dir + "/rgb");
  if (all_images.empty())
    all_images = list_images(dataset_dir);
  std::vector<RobotPoseRecord> all_records;
  try {
    all_records = load_robot_poses(dataset_dir + "/poses.csv");
  } catch (const std::exception &e) {
    std::cerr << "Failed to load robot poses: " << e.what()
              << "\nDid you run `capture --dataset " << dataset_dir
              << " --base-frame ...` first?\n";
    return 2;
  }

  if (all_images.size() != all_records.size()) {
    std::cerr << "Need exactly one pose row per image; images="
              << all_images.size() << " poses=" << all_records.size() << "\n";
    return 2;
  }
  for (size_t i = 0; i < all_images.size(); ++i) {
    if (basename_of(all_images[i]) != all_records[i].image_name) {
      std::cerr << "poses.csv row " << i << " names '"
                << all_records[i].image_name
                << "' but the sorted image list has '"
                << basename_of(all_images[i])
                << "' at that position -- refusing to guess the pairing.\n";
      return 2;
    }
  }
  if (all_records.size() < static_cast<size_t>(5 + validate_count)) {
    std::cerr << "Need at least 5 calibration samples plus "
              << validate_count << " held-out validation sample(s); only "
              << all_records.size() << " captured.\n";
    return 2;
  }

  // The last `validate_count` captures never touch intrinsic refinement or
  // the hand-eye solve -- they exist solely to check the solved matrix
  // against poses/images it has never seen (requirement: independent
  // validation).
  size_t n_calib = all_records.size() - validate_count;
  std::vector<std::string> calib_images(all_images.begin(),
                                        all_images.begin() + n_calib);
  std::vector<Pose3D> calib_poses;
  for (size_t i = 0; i < n_calib; ++i)
    calib_poses.push_back(all_records[i].pose);

  std::vector<std::string> validation_images(all_images.begin() + n_calib,
                                             all_images.end());
  std::vector<RobotPoseRecord> validation_records(all_records.begin() +
                                                       n_calib,
                                                   all_records.end());

  Logger::section("Category 1: Intrinsic Calibration");
  Logger::info("Using " + std::to_string(calib_images.size()) +
              " images for calibration, " +
              std::to_string(validation_images.size()) +
              " held out for validation.");

  IntrinsicCalibrator calib_obj;
  if (!calib_obj.intrinsicCalibrate(cfg, calib_images)) {
    Logger::error("Intrinsic calibration failed!");
    return -1;
  }
  Logger::success("Intrinsic calibration success!");
#endif

#if RUN_HAND_EYE_3D_BOARD_IMG
  Logger::section(
      "Category 2: 3D Camera Hand-Eye Calibration - Based on Board Image");

  Logger::info("Starting hand-eye calibration...");
  BoardImageCalibrator board_calib_obj;
  board_calib_obj.setConfig(cfg);
  board_calib_obj.setImageFiles(calib_images);
  board_calib_obj.setRobotPoses(calib_poses);
  board_calib_obj.setCalibrationType(cfg.calib_type);
  // Robot poses come straight from a TF quaternion via
  // tf2::Matrix3x3::getRPY(), which *is* extrinsic (fixed-axis) X-Y-Z --
  // the one Euler convention this project's RPY::toRotation3D() actually
  // implements, and the same convention URDF/ROS <origin rpy="..."/> uses.
  board_calib_obj.setRPYType(RPY::RPYType::XYZ, RPY::ReferenceType::EXTRINSIC);

  if (!board_calib_obj.runCalibration()) {
    Logger::error("3D Hand-Eye Calibration failed!");
    return -1;
  }
  Logger::success("3D Hand-Eye Calibration success!");
  std::cout << "\n";
  Pose3D hand_eye_pose = board_calib_obj.getResultPose();

  // hand_eye_pose comes from cv::calibrateHandEye() (PARK), which properly
  // decouples rotation and translation instead of inverting one big coupled
  // matrix -- it no longer explodes to meters of error on marginal data the
  // way this project's original custom closed-form solver did. But a
  // numerically well-behaved answer isn't automatically a *correct* one:
  // when the robot's own rotation doesn't excite enough independent
  // directions (e.g. a pan/tilt head with no roll, poses clustered near
  // neutral), different, genuinely independent algorithms can still land
  // on substantially different translations from the very same images.
  // Two cheap tripwires, neither of which requires understanding the
  // solver's internals:
  constexpr double kMaxPlausibleOffsetMm = 300.0;
  constexpr double kMaxCrossCheckDiscrepancyMm = 30.0;
  {
    // 1) A camera bolted directly to a robot link is never offset by more
    // than ~30cm from that link's origin in any real design.
    double offset_mm = hand_eye_pose.xyz().eigen().norm();
    if (offset_mm > kMaxPlausibleOffsetMm) {
      Logger::warn(
          "hand_eye translation magnitude is " + std::to_string(offset_mm) +
          "mm -- implausible for a camera mounted on this link. Do not "
          "trust this result -- recapture with a wider, more varied "
          "yaw/pitch sweep.");
    }
    // 2) PARK (used above) and TSAI solve rotation and translation via
    // genuinely different math; on well-supported data they agree closely.
    // A big gap here means the sample set doesn't carry enough information
    // to pin down a unique answer, even though neither number individually
    // looks obviously wrong.
    double cross_check_mm = board_calib_obj.getCrossCheckTranslationDiscrepancyMm();
    if (cross_check_mm >= 0 && cross_check_mm > kMaxCrossCheckDiscrepancyMm) {
      Logger::warn(
          "PARK vs. TSAI hand-eye translation disagree by " +
          std::to_string(cross_check_mm) +
          "mm on the same images. Neither individual answer may look "
          "wrong, but this disagreement means the sample set doesn't "
          "carry enough independent rotation information to determine a "
          "unique, trustworthy result -- treat the result below with "
          "suspicion and prefer recapturing with more varied poses over "
          "trusting either number.");
    }
  }
  auto errors = board_calib_obj.getErrors();
  auto markers = board_calib_obj.getMarkerPoints();
  auto marker_success = board_calib_obj.getMarkerSuccess();

  std::cout << "[INFO] Index | Robot(x,y,z,rx,ry,rz) | Marker(x,y,z) | Error\n";
  std::cout
      << "--------------------------------------------------------------\n";
  for (size_t i = 0; i < calib_poses.size(); ++i) {
    Pose3D p = calib_poses[i];
    std::cout << std::setw(3) << i << " | " << std::setw(10) << p.x() << ", "
              << std::setw(10) << p.y() << ", " << std::setw(10) << p.z()
              << ", " << std::setw(10) << p.rx() << ", " << std::setw(10)
              << p.ry() << ", " << std::setw(10) << p.rz() << " | ";

    if (i < marker_success.size() && marker_success[i]) {
      std::cout << std::setw(10) << markers[i].x() << ", " << std::setw(10)
                << markers[i].y() << ", " << std::setw(10) << markers[i].z();
    } else {
      std::cout << std::setw(10) << "N/A" << ", " << std::setw(10) << "N/A"
                << ", " << std::setw(10) << "N/A";
    }

    std::cout << " | ";
    if (i < errors.size() && errors[i] >= 0)
      std::cout << std::setw(10) << errors[i];
    else
      std::cout << std::setw(10) << "N/A";
    std::cout << "\n";
  }

  ErrorStats fit_stats = compute_error_stats(errors);
  Logger::success("3D Hand-Eye Calibration successful! RMS: " +
                  std::to_string(fit_stats.rms) +
                  " mm, Avg: " + std::to_string(fit_stats.avg) +
                  " mm, Max: " + std::to_string(fit_stats.max) + " mm (n=" +
                  std::to_string(fit_stats.count) + ")");

  std::cout << "[INFO] T_head_pitch_Link_camera_optical -- maps a point p_cam\n"
               "       in the camera optical frame into head_pitch_Link via\n"
               "       p_head_pitch_Link = R * p_cam + t (Eigen 4x4 form):\n"
            << hand_eye_pose.toTransform3D().eigen() << "\n"
            << std::endl;

  std::cout << "[RESULT] hand_eyes (Pose3D format): \n"
            << hand_eye_pose.x() << ", " << hand_eye_pose.y() << ", "
            << hand_eye_pose.z() << ", " << hand_eye_pose.rx() << ", "
            << hand_eye_pose.ry() << ", " << hand_eye_pose.rz() << std::endl;

  // --- Held-out validation: samples in validation_images/validation_records
  // never participated in intrinsic refinement or the hand-eye solve above.
  cv::Mat K_first, dist_first;
  {
    cv::FileStorage fs_1st(cfg.xml_intrinsic_1st, cv::FileStorage::READ);
    fs_1st["K"] >> K_first;
    fs_1st["distortion"] >> dist_first;
  }
  cv::Mat K_second, dist_second;
  board_calib_obj.getRefinedIntrinsics(K_second, dist_second);
  Vector3D fixed_point_base = board_calib_obj.getFixedPoint();

  std::vector<double> validation_errors;
  if (!validation_images.empty()) {
    Logger::section("Category 2b: Independent Pose Validation (held-out)");
    std::cout
        << "[INFO] Index | Image | Predicted(x,y,z in capture's --base-frame) "
           "| Error\n";
    std::cout
        << "--------------------------------------------------------------\n";
    for (size_t i = 0; i < validation_images.size(); ++i) {
      ValidationResult v = validate_held_out_sample(
          validation_images[i], validation_records[i].pose, cfg, K_first,
          dist_first, K_second, dist_second, hand_eye_pose, fixed_point_base);
      std::cout << std::setw(3) << i << " | "
                << basename_of(validation_images[i]) << " | ";
      if (v.board_found) {
        std::cout << std::setw(10) << v.predicted_point_base_frame.x() << ", "
                  << std::setw(10) << v.predicted_point_base_frame.y() << ", "
                  << std::setw(10) << v.predicted_point_base_frame.z()
                  << " | " << std::setw(10) << v.error_mm;
        validation_errors.push_back(v.error_mm);
      } else {
        std::cout << std::setw(10) << "N/A" << ", " << std::setw(10) << "N/A"
                  << ", " << std::setw(10) << "N/A" << " | " << std::setw(10)
                  << "N/A";
        validation_errors.push_back(-1.0);
      }
      std::cout << "\n";
    }
    ErrorStats val_stats = compute_error_stats(validation_errors);
    if (val_stats.count > 0) {
      Logger::success(
          "Independent validation: RMS: " + std::to_string(val_stats.rms) +
          " mm, Avg: " + std::to_string(val_stats.avg) +
          " mm, Max: " + std::to_string(val_stats.max) + " mm (n=" +
          std::to_string(val_stats.count) + ")");
    } else {
      Logger::warn("Independent validation: board not detected in any "
                  "held-out sample.");
    }
  } else {
    Logger::warn("--validate-count 0: no independent pose validation run.");
  }

  // The URDF models the camera mount as TWO joints:
  //   head_pitch_Link --(head_d435i_joint, the one to calibrate)--> head_d435i_link
  //   head_d435i_link --(head_d435i_optical_joint, FIXED, rpy="-1.5708 0 -1.5708")--> head_d435i_optical_frame
  // solvePnP/OpenCV work in the optical convention, so hand_eye_pose above
  // is T_head_pitch_Link_camera_optical -- correct for the *_optical_frame
  // child, but NOT what belongs in head_d435i_joint (link convention). Pasting
  // the optical-convention numbers into head_d435i_joint directly is exactly
  // the mistake this section exists to prevent: same translation magnitude
  // (both frames share an origin) but a rotated/wrong-axis result, easy to
  // miss without a numeric check.
  // R_link_from_optical: p_link_local = R_lo * p_optical_local (from the
  // fixed joint's rpy="-1.5708 0 -1.5708", extrinsic X-Y-Z).
  Eigen::Matrix3d R_link_from_optical;
  R_link_from_optical << 0, 0, 1,
                         -1, 0, 0,
                          0, -1, 0;
  Eigen::Matrix4d T_optical = hand_eye_pose.toTransform3D().eigen();
  Eigen::Matrix4d T_link = Eigen::Matrix4d::Identity();
  T_link.block<3, 3>(0, 0) =
      T_optical.block<3, 3>(0, 0) * R_link_from_optical.transpose();
  T_link.block<3, 1>(0, 3) = T_optical.block<3, 1>(0, 3); // shared origin
  Pose3D hand_eye_pose_link = Pose3D(Transform3D(T_link));

  std::cout << "[INFO] T_head_pitch_Link_" << child_frame
            << "_LINK (paste into head_d435i_joint's <origin>, NOT the "
               "*_optical block above):\n"
            << "  xyz(mm) = " << hand_eye_pose_link.x() << ", "
            << hand_eye_pose_link.y() << ", " << hand_eye_pose_link.z()
            << "\n  rpy(deg) = " << hand_eye_pose_link.rx() << ", "
            << hand_eye_pose_link.ry() << ", " << hand_eye_pose_link.rz()
            << std::endl;

  cv::Mat H_hand_eyes =
      CoordinateTransformer::toHomogeneousMatrix(hand_eye_pose);
  cv::Mat H_cam_to_gripper_R = H_hand_eyes(cv::Rect(0, 0, 3, 3)).clone();
  cv::Mat H_cam_to_gripper_t = H_hand_eyes(cv::Rect(3, 0, 1, 3)).clone();

  Logger::info("Hand-eye calibration results (R | t) saved to: " +
               cfg.xml_extrinsic);

  cv::FileStorage fs_hand_eye(cfg.xml_extrinsic, cv::FileStorage::WRITE);
  fs_hand_eye << "R" << H_cam_to_gripper_R << "t" << H_cam_to_gripper_t;
  fs_hand_eye << "calibration_type" << "EIH"
              << "translation_unit" << "mm"
              << "matrix_semantics" << "T_head_pitch_Link_camera_optical"
              << "matrix_convention"
              << "p_head_pitch_Link = R * p_camera_optical + t"
              << "link_convention_xyz_mm"
              << (cv::Mat_<double>(1, 3) << hand_eye_pose_link.x(),
                  hand_eye_pose_link.y(), hand_eye_pose_link.z())
              << "link_convention_rpy_deg"
              << (cv::Mat_<double>(1, 3) << hand_eye_pose_link.rx(),
                  hand_eye_pose_link.ry(), hand_eye_pose_link.rz())
              << "link_convention_note"
              << "head_pitch_Link -> head_d435i_link (fixed joint), paste "
                 "this xyz/rpy into head_d435i_joint's <origin>"
              << "board_squares" << "[12, 9]"
              << "square_size_mm" << 15 << "fit_rms_mm" << fit_stats.rms
              << "fit_avg_mm" << fit_stats.avg << "fit_max_mm" << fit_stats.max
              << "fit_sample_count" << (int)fit_stats.count;
  if (!validation_errors.empty()) {
    ErrorStats val_stats = compute_error_stats(validation_errors);
    fs_hand_eye << "validation_rms_mm" << val_stats.rms << "validation_avg_mm"
                << val_stats.avg << "validation_max_mm" << val_stats.max
                << "validation_sample_count" << (int)val_stats.count;
  }
  fs_hand_eye.release();
  write_urdf_snippet(output_dir + "/head_d435i_optical.urdf.xml",
                     hand_eye_pose, "head_pitch_Link", child_frame);
  write_urdf_snippet(output_dir + "/head_d435i_link.urdf.xml",
                     hand_eye_pose_link, "head_pitch_Link", "head_d435i_link",
                     "child *link* frame (ROS REP-103 convention: X "
                     "forward, Y left, Z up -- NOT the optical convention; "
                     "paste this straight into head_d435i_joint)");
  Logger::success("3D Hand-Eye Calibration complete!");
#endif

#if RUN_FOUR_POINT_2D
  Logger::section("Category 3: 2D Camera Hand-Eye Calibration - 4-Point Pin");

  std::vector<cv::Point2d> img_points = {
      {2035, 1093}, {3129, 1087}, {3130, 1549}, {2038, 1556}};
  std::vector<cv::Point2d> robot_points = {
      {800.94, 886.14}, {805.45, 570.15}, {671.78, 568.56}, {667.07, 884.8}};
  if (img_points.size() < 4 || robot_points.size() < 4)
    return 1;

  FourPointCalibrator fp_calib;
  cv::Mat hand_eyes_4p;
  if (!fp_calib.calibrate(img_points, robot_points, hand_eyes_4p)) {
    Logger::error("Four-point calibration failed!");
    return -1;
  }
  std::cout << "[INFO] hand_eyes Homogeneous Matrix (2x3):\n"
            << hand_eyes_4p << "\n"
            << std::endl;

  cv::Point2d test_img_pt = {2035, 1093};
  cv::Point2d robot_point = fp_calib.transform(test_img_pt, hand_eyes_4p);

  std::cout << "\033[1;32m[VERIFY] Four-point pin verification: Image("
            << test_img_pt.x << "," << test_img_pt.y << ") → Robot("
            << robot_point.x << ", " << robot_point.y << ")\033[0m\n"
            << std::endl;
#endif

#if RUN_TWELVE_POINT_2D
  Logger::section("Category 4: 2D Camera Hand-Eye Calibration - 12-Point");
  TwelvePointCalibrator calib12p_obj;
  TwelvePointCalibrator::CalibrationType2D hand_eye_type =
      TwelvePointCalibrator::CalibrationType2D::EIH;
  TwelvePointCalibrator::CameraInstallType install_type =
      TwelvePointCalibrator::CameraInstallType::SameToTCPZ;
  Pose2D registration_point = {1188, 468, 0};
  std::vector<Pose2D> img_points_12p = {
      {1188, 468, -0}, {1173, 467, -0}, {1195, 464, -0}, {1320, 474, -0},
      {1249, 593, 5},  {1292, 545, 7},  {1346, 632, 6},  {1439, 520, 5},
      {1482, 595, 4},  {1329, 437, 6},  {1239, 472, -0}, {1420, 550, -14}};
  std::vector<Pose2D> robot_poses_12p = {
      {0, 0, 0},    {0, 30, 0},  {30, 30, 0},   {30, 0, 0},
      {30, -30, 0}, {0, -30, 0}, {-30, -30, 0}, {-30, 0, 0},
      {-30, 30, 0}, {0, 0, -10}, {0, 0, 0},     {0, 0, 10}};
  calib12p_obj.setHandEyeType(hand_eye_type);
  calib12p_obj.setCameraInstallType(install_type);
  calib12p_obj.setRegistrationPoint(registration_point);
  calib12p_obj.setImageMarksPoints(img_points_12p);
  calib12p_obj.setRobotPoses(robot_poses_12p);

  calib12p_obj.runCalibration();

  auto errors_12p = calib12p_obj.getCalibrationErrors();
  std::cout << "[INFO] Index | Robot(dx,dy,rz) | Image(x,y,theta) | Error\n";
  std::cout << "---------------------------------------------------------"
               "---------------------\n";
  for (size_t i = 0; i < robot_poses_12p.size(); ++i) {
    Pose2D p_robot = robot_poses_12p[i];
    Pose2D p_img = img_points_12p[i];
    std::cout << std::setw(3) << i << " | " << std::setw(10) << p_robot.x()
              << ", " << std::setw(10) << p_robot.y() << ", " << std::setw(10)
              << p_robot.angle() << " | " << std::setw(10) << p_img.x() << ", "
              << std::setw(10) << p_img.y() << ", " << std::setw(10)
              << p_img.angle() << " | ";
    if (i < errors_12p.size())
      std::cout << std::setw(10) << errors_12p[i];
    else
      std::cout << std::setw(10) << "N/A";
    std::cout << "\n";
  }

  Logger::success("2D Twelve-point Calibration successful!");

  auto trans_pose_12p = calib12p_obj.getTransformPose();
  std::cout << "[INFO] Transform Pose (2x3 Affine Matrix):\n"
            << trans_pose_12p(0, 0) << ", " << trans_pose_12p(0, 1) << ", "
            << trans_pose_12p(0, 2) << "\n"
            << trans_pose_12p(1, 0) << ", " << trans_pose_12p(1, 1) << ", "
            << trans_pose_12p(1, 2) << "\n"
            << std::endl;

  auto raw_tool_center = calib12p_obj.getRawToolCenter();
  auto fitting_radius = calib12p_obj.getFittingRadius();
  std::cout << "[INFO] Circle Fitting Result: \n"
            << "center=(" << raw_tool_center[0] << ", " << raw_tool_center[1]
            << "), R=(" << fitting_radius << ")\n"
            << std::endl;

  auto tcp_center = calib12p_obj.getRotationCenter();
  std::cout << "[RESULT] TCP Rotation Center (relative to registration point): "
            << tcp_center[0] << ", " << tcp_center[1] << std::endl;

  Pose2D test_img = {1188, 468, 0};
  Pose2D register_point = calib12p_obj.getRegistrationPoint();
  Vector2d tool_center = calib12p_obj.getRotationCenter();
  Matrix<double, 2, 3> trans_pose = calib12p_obj.getTransformPose();
  Pose2D robot_out;
  calib12p_obj.transformImageToRobotPose(test_img, register_point, tool_center,
                                         trans_pose, robot_out);
  std::cout << "\033[1;32m[VERIFY] Twelve-point verification: Image("
            << test_img.x() << "," << test_img.y() << ") → Robot("
            << robot_out.x() << ", " << robot_out.y() << ", "
            << robot_out.angle() << "°)\033[0m\n"
            << std::endl;
#endif

#if RUN_HAND_EYE_3D_BALL
  Logger::section(
      "Category 5: 3D Camera Hand-Eye Calibration - Based on Ball Center XYZ");

  // Robot poses: x, y, z, rx, ry, rz
  std::vector<std::vector<double>> robot_poses_data = {
      {-1223.07, 1493.14, 774.351, -172.691, -14.074, 170.725},
      {-1105.25, 1370.07, 774.303, -171.213, -18.84, 153.076},
      {-1064.56, 1424.31, 808.086, 166.736, -18.035, 160.874},
      {-1154.64, 1396.67, 780.382, 168.416, -13.818, 164.482},
      {-1099.28, 1471.44, 637.932, -170.201, -0.875, 160.177},
      {-1215.18, 1368.72, 656.108, -173.238, 14.414, 173.631},
      {-1148.45, 1422.8, 730.753, -176.372, 10.735, 163.365},
      {-1132.76, 1358.11, 775.324, 178.924, 4.649, 171.376},
      {-1231.24, 1314.03, 762.005, 169.6, -10.535, 173.234},
      {-1186.72, 1408.22, 682.34, 174.96, -4.641, 165.91},
      {-1106.26, 1441.8, 686.117, 179.956, 0.57, 152.826}};

  // Marker points (pixel coordinates): x, y, z
  std::vector<std::vector<double>> marker_points_data = {
      {-107, -2.25, 632.75},   {-3.75, -39.25, 528},
      {48.75, -37, 627.5},     {-29.75, -14.25, 580.25},
      {46.75, 108.75, 584.5},  {-19.5, 59.75, 490.75},
      {21, 2.5, 563},          {38.75, -47.25, 525.75},
      {-89.25, -16.75, 487.5}, {-36, 78.25, 554.25},
      {39.25, 65.5, 575.5}};

  // Prepare data for calibration
  std::vector<Eigen::Vector3d> mark_points;
  std::vector<Pose3D> robot_poses_p3d;

  for (size_t i = 0; i < robot_poses_data.size(); ++i) {
    const auto &robot = robot_poses_data[i];
    const auto &marker = marker_points_data[i];

    robot_poses_p3d.push_back(
        Pose3D(robot[0], robot[1], robot[2], robot[3], robot[4], robot[5]));
    mark_points.push_back(Eigen::Vector3d(marker[0], marker[1], marker[2]));
  }

  // Perform calibration using BallCalibrator object
  BallCalibrator ball_calib;
  ball_calib.setRobotPoses(robot_poses_p3d);
  ball_calib.setMarkPoints(mark_points);
  ball_calib.setCalibrationType(CalibrationType3D::ETH);
  ball_calib.setRPYType(RPY::RPYType::XYZ, RPY::ReferenceType::EXTRINSIC);

  if (ball_calib.runCalibration()) {
    Pose3D result = ball_calib.getResultPose();
    auto errors = ball_calib.getErrors();

    std::cout
        << "[INFO] Index | Robot(x,y,z,rx,ry,rz) | Marker(x,y,z) | Error\n";
    std::cout
        << "--------------------------------------------------------------\n";
    for (size_t i = 0; i < mark_points.size(); ++i) {
      Pose3D p = robot_poses_p3d[i];
      std::cout << std::setw(3) << i << " | " << std::setw(10) << p.x() << ", "
                << std::setw(10) << p.y() << ", " << std::setw(10) << p.z()
                << ", " << std::setw(10) << p.rx() << ", " << std::setw(10)
                << p.ry() << ", " << std::setw(10) << p.rz() << " | "
                << std::setw(10) << mark_points[i].x() << ", " << std::setw(10)
                << mark_points[i].y() << ", " << std::setw(10)
                << mark_points[i].z() << " | ";
      if (i < errors.size() && errors[i] >= 0)
        std::cout << std::setw(10) << errors[i];
      else
        std::cout << std::setw(10) << "N/A";
      std::cout << "\n";
    }

    double avg_error = 0;
    for (auto e : errors)
      avg_error += e;
    avg_error /= errors.size();

    Logger::success("3D Ball-based Calibration successful! Avg Error: " +
                    std::to_string(avg_error));

    std::cout << "[INFO] hand_eyes Homogeneous Matrix (Eigen format):\n"
              << result.toTransform3D().eigen() << "\n"
              << std::endl;

    std::cout << "[RESULT] hand_eyes (Pose3D format): \n"
              << result.x() << ", " << result.y() << ", " << result.z() << ", "
              << result.rx() << ", " << result.ry() << ", " << result.rz()
              << std::endl;
  } else {
    Logger::warn("3D Ball-based Calibration failed.");
  }
#endif

#if RUN_MULTI_POINT_3D
  /** 多点 3D 仿射：机器人位姿仅用到 (x,y,z) + 相机系标定球心 (x,y,z) */
  const std::vector<std::vector<double>> robot_pose = {
      {-621.59, 1305.31, 1285.92},
      {-413.87, 1497.00, 1046.40},
      {-169.71, 1531.33, 1128.30},
      {-340.03, 1630.08, 1071.06},
      {-602.10, 1768.01, 922.36},
      {-248.67, 1830.81, 979.31},
  };

  const std::vector<std::vector<double>> marker_xyz = {
    {-243.55, -199.25, 561.592},
    {-49.562, 102.437, 645.957},
    {210.057, 58.6,   728.596},
    {7.5,    151.933, 778.472},
    {-268.88, 336.168, 811.641},
    {87.0192, 331.325, 923.076}
  };

  Logger::section(
      "Category 6: 3D xyz–xyz Affine Fit ");

  if (robot_pose.size() != marker_xyz.size() ||
      robot_pose.size() < 4) {
    Logger::error("3D xyz–xyz demo data: need ≥4 matching pairs.");
    return -1;
  }

  std::vector<cv::Point3d> marker_pts_3d;
  std::vector<cv::Point3d> robot_pts_3d;
  marker_pts_3d.reserve(robot_pose.size());
  robot_pts_3d.reserve(robot_pose.size());
  for (size_t i = 0; i < robot_pose.size(); ++i) {
    const auto &r = robot_pose[i];
    const auto &m = marker_xyz[i];
    robot_pts_3d.emplace_back(r[0], r[1], r[2]);
    marker_pts_3d.emplace_back(m[0], m[1], m[2]);
  }

  MultiPoint3DCalibrator mp3d;
  cv::Mat hand_eyes_3d_4p;
  if (!mp3d.calibrate(marker_pts_3d, robot_pts_3d, hand_eyes_3d_4p)) {
    Logger::error("3D xyz–xyz affine calibration failed!");
    return -1;
  }

  std::cout << "[INFO] 3×4 affine M (robot ≈ M * [marker;1]):\n"
            << hand_eyes_3d_4p << "\n"
            << std::endl;

  std::cout << "[INFO] Index | Robot(x,y,z) | Marker(x,y,z) | Pred(x,y,z) | "
               "|Δ||\n";
  std::cout << "------------------------------------------------------------"
               "------------------\n";
  const auto &perr = mp3d.getPerPointErrors();
  for (size_t i = 0; i < robot_pts_3d.size(); ++i) {
    cv::Point3d pr = mp3d.transform(marker_pts_3d[i], hand_eyes_3d_4p);
    std::cout << std::setw(3) << i << " | " << std::setw(10) << robot_pts_3d[i].x
              << ", " << std::setw(10) << robot_pts_3d[i].y << ", "
              << std::setw(10) << robot_pts_3d[i].z << " | " << std::setw(10)
              << marker_pts_3d[i].x << ", " << std::setw(10)
              << marker_pts_3d[i].y << ", " << std::setw(10)
              << marker_pts_3d[i].z << " | " << std::setw(10) << pr.x << ", "
              << std::setw(10) << pr.y << ", " << std::setw(10) << pr.z
              << " | ";
    if (i < perr.size())
      std::cout << std::setw(10) << perr[i];
    else
      std::cout << std::setw(10) << "N/A";
    std::cout << "\n";
  }

  Logger::success("3D xyz–xyz affine fit RMSE: " +
                  std::to_string(mp3d.getRmse()));

  cv::Point3d test_mk = marker_pts_3d.front();
  cv::Point3d test_rb = mp3d.transform(test_mk, hand_eyes_3d_4p);
  std::cout << "\033[1;32m[VERIFY] 3D affine: Marker(" << test_mk.x << ","
            << test_mk.y << "," << test_mk.z << ") → Robot(" << test_rb.x
            << "," << test_rb.y << "," << test_rb.z << ")\033[0m\n"
            << std::endl;
#endif


#if RUN_HAND_EYE_3D_BOARD_CLOUD
  Logger::section("Category 6: 3D Camera Hand-Eye Calibration - Based on Board "
                  "Image + Cloud");
  std::vector<std::string> img_files = {"../assert/1.png", "../assert/2.png",
                                        "../assert/3.png", "../assert/4.png",
                                        "../assert/5.png"};
  std::vector<std::string> cloud_luts_files = {
      "../assert/lut/1.bin", "../assert/lut/2.bin", "../assert/lut/3.bin",
      "../assert/lut/4.bin", "../assert/lut/5.bin"};
  std::vector<std::string> cloud_pcd_files = {
      "../assert/cloud/1.pcd", "../assert/cloud/2.pcd", "../assert/cloud/3.pcd",
      "../assert/cloud/4.pcd", "../assert/cloud/5.pcd"};

  BoardCloudCalibrator cloud_calib;
  cloud_calib.setConfig(cfg);
  cloud_calib.setImageFiles(img_files);
  cloud_calib.setCloudLuts(cloud_luts_files);
  cloud_calib.setCloudFiles(cloud_pcd_files);
  cloud_calib.setRobotPoses(robot_poses);
  cloud_calib.setCalibrationType(cfg.calib_type);
  cloud_calib.setRPYType(RPY::RPYType::XYZ, RPY::ReferenceType::EXTRINSIC);

  if (cloud_calib.runCalibration()) {
    Pose3D result = cloud_calib.getResultPose();
    auto errors = cloud_calib.getErrors();
    auto markers = cloud_calib.getMarkerPoints();
    auto marker_success = cloud_calib.getMarkerSuccess();

    std::cout
        << "[INFO] Index | Robot(x,y,z,rx,ry,rz) | Marker(x,y,z) | Error\n";
    std::cout
        << "--------------------------------------------------------------\n";
    for (size_t i = 0; i < robot_poses.size(); ++i) {
      Pose3D p = robot_poses[i];
      std::cout << std::setw(3) << i << " | " << std::setw(10) << p.x() << ", "
                << std::setw(10) << p.y() << ", " << std::setw(10) << p.z()
                << ", " << std::setw(10) << p.rx() << ", " << std::setw(10)
                << p.ry() << ", " << std::setw(10) << p.rz() << " | ";

      if (i < marker_success.size() && marker_success[i]) {
        std::cout << std::setw(10) << markers[i].x() << ", " << std::setw(10)
                  << markers[i].y() << ", " << std::setw(10) << markers[i].z();
      } else {
        std::cout << std::setw(10) << "N/A" << ", " << std::setw(10) << "N/A"
                  << ", " << std::setw(10) << "N/A";
      }

      std::cout << " | ";
      if (i < errors.size() && errors[i] >= 0)
        std::cout << std::setw(10) << errors[i];
      else
        std::cout << std::setw(10) << "N/A";
      std::cout << "\n";
    }

    double errorSum = 0.0;
    int errCount = 0;
    for (auto error : errors) {
      if (error >= 0) {
        errorSum += error;
        errCount++;
      }
    }
    double avg = errCount == 0 ? 0 : errorSum / errCount;

    double maxE = 0, minE = 0;
    bool first = true;
    for (auto error : errors) {
      if (error < 0)
        continue;
      if (first) {
        maxE = minE = error;
        first = false;
      } else {
        maxE = std::max(maxE, error);
        minE = std::min(minE, error);
      }
    }

    Logger::success("3D Board Cloud Calibration successful! Avg Error: " +
                    std::to_string(avg) + ", Max: " + std::to_string(maxE) +
                    ", Min: " + std::to_string(minE));

    std::cout << "[INFO] hand_eyes Homogeneous Matrix (Eigen format):\n"
              << result.toTransform3D().eigen() << "\n"
              << std::endl;

    std::cout << "[RESULT] hand_eyes (Pose3D format): \n"
              << result.x() << ", " << result.y() << ", " << result.z() << ", "
              << result.rx() << ", " << result.ry() << ", " << result.rz()
              << std::endl;
  } else {
    Logger::warn("3D Board Cloud Calibration failed.");
  }
#endif

  Logger::section("All Calibrations Complete!");

  return 0;
}
