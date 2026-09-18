#include "calibration/05_hand_eye_3d_board_img/BoardImageCalibrator.h"
#include "common/CoordinateTransformer.h"
#include <algorithm>
#include <iostream>
#include <opencv2/core/eigen.hpp>

Pose3D BoardImageCalibrator::extrinsicCalibrate(
    const CalibConfig &cfg, const std::vector<std::string> imgs,
    const std::vector<Pose3D> &robot_poses, CalibrationType3D calib_type) {
  setConfig(cfg);
  setImageFiles(imgs);
  setRobotPoses(robot_poses);
  setCalibrationType(calib_type);

  if (runCalibration()) {
    return result_pose_;
  }
  return Pose3D();
}

bool BoardImageCalibrator::runCalibration() {
  cv::FileStorage fs(cfg_.xml_intrinsic_1st, cv::FileStorage::READ);
  cv::Mat K, dist;
  fs["K"] >> K;
  fs["distortion"] >> dist;
  fs.release();
  if (K.empty()) {
    std::cerr << "未找到内参！请先运行 intrinsicCalibrate\n";
    return false;
  }
  std::vector<cv::Mat> all_images;
  std::vector<std::vector<cv::Point3f>> worlds_list_second;
  std::vector<std::vector<cv::Point2f>> pixels_list_second;
  cv::Size imgSize;
  int valid = 0;
  for (size_t i = 0; i < img_files_.size(); ++i) {
    cv::Mat raw = cv::imread(img_files_[i]);
    if (raw.empty())
      continue;
    cv::Mat input_image_in, input_image_out, output_corners_image;
    input_image_in = raw.clone();
    cv::undistort(input_image_in, input_image_out, K, dist);
    all_images.push_back(input_image_out);
    if (imgSize == cv::Size())
      imgSize = raw.size();
    std::vector<cv::Point2f> pixel_corners;
    std::vector<cv::Point3f> world_corners;
    int ret = IntrinsicCalibrator::detectCalibBoard(
        input_image_out, output_corners_image, pixel_corners, world_corners,
        cfg_, 0);
    if (ret == 1) {
      worlds_list_second.push_back(world_corners);
      pixels_list_second.push_back(pixel_corners);
      valid++;
    }
    std::cout << "\r进度: " << i + 1 << "/" << img_files_.size()
              << " 有效: " << valid << std::flush;
  }
  std::cout << "\n";
  if (valid < 5) {
    std::cerr << "至少 5 张！\n";
    return false;
  }
  cv::Mat K_second, dis_second;
  if (cfg_.use_fixed_intrinsics && !cfg_.fixed_K.empty()) {
    K_second = cfg_.fixed_K.clone();
    dis_second = cfg_.fixed_dist.clone();
    std::cout << "使用 RealSense 实测内参，跳过二次内参拟合。\n";
    std::cout << "内参: \n" << K_second << "\n";
    std::cout << "畸变: \n" << dis_second << "\n";
  } else {
    std::vector<cv::Mat> rvecs, tvecs;
    double rms =
        cv::calibrateCamera(worlds_list_second, pixels_list_second, imgSize,
                            K_second, dis_second, rvecs, tvecs, 0);
    std::cout << "第二次标定成功！RMS = " << rms << "\n";
    std::cout << "内参: \n" << K_second << "\n";
    std::cout << "畸变: \n" << dis_second << "\n";
  }
  cv::FileStorage fs_2nd(cfg_.xml_intrinsic_2nd, cv::FileStorage::WRITE);
  fs_2nd << "K" << K_second << "distortion" << dis_second << "pattern"
         << (int)cfg_.pattern;
  fs_2nd.release();
  refined_K_ = K_second.clone();
  refined_dist_ = dis_second.clone();

  // Per-sample robot pose (gripper2base) and full board pose (target2cam,
  // rotation *and* translation -- solvePnP's rvec was previously discarded,
  // keeping only tvec). cv::calibrateHandEye() (Tsai/Park/Horaud) needs both
  // and, unlike this project's original point-only closed-form solver,
  // solves rotation and translation as two separate, well-conditioned
  // steps instead of one coupled 15-unknown linear system -- verified far
  // more robust when the robot's own rotation only samples 2 DOF (e.g. a
  // pan/tilt head with no roll): the old solver could blow up to meters of
  // error on marginal data, calibrateHandEye stayed within a few cm on the
  // exact same images.
  std::vector<cv::Mat> Rs_gripper2base, ts_gripper2base;
  std::vector<cv::Mat> Rs_target2cam, ts_target2cam;
  std::vector<Pose3D> success_poses;
  marker_points_.assign(robot_poses_.size(), Vector3D(0, 0, 0));
  marker_success_.assign(robot_poses_.size(), false);

  for (size_t i = 0; i < robot_poses_.size(); ++i) {
    if (i >= all_images.size())
      continue;

    cv::Mat output_corners_image;
    std::vector<cv::Point2f> pixel_corners;
    std::vector<cv::Point3f> world_corners;
    int ret = IntrinsicCalibrator::detectCalibBoard(
        all_images[i], output_corners_image, pixel_corners, world_corners, cfg_,
        1);
    if (ret == 1) {
      cv::Mat rvec, tvec;
      cv::solvePnP(world_corners, pixel_corners, K_second, dis_second, rvec,
                   tvec);
      cv::Mat R_target2cam;
      cv::Rodrigues(rvec, R_target2cam);

      Vector3D point_xyz_camera;
      point_xyz_camera[0] = tvec.at<double>(0);
      point_xyz_camera[1] = tvec.at<double>(1);
      point_xyz_camera[2] = tvec.at<double>(2);
      marker_points_[i] = point_xyz_camera;
      marker_success_[i] = true;

      Pose3D pose_copy = robot_poses_[i]; // toHomogeneousMatrix() takes a
                                          // non-const ref
      cv::Mat H = CoordinateTransformer::toHomogeneousMatrix(pose_copy);
      Rs_gripper2base.push_back(H(cv::Rect(0, 0, 3, 3)).clone());
      ts_gripper2base.push_back(H(cv::Rect(3, 0, 1, 3)).clone());
      Rs_target2cam.push_back(R_target2cam);
      ts_target2cam.push_back(tvec.clone());
      success_poses.push_back(robot_poses_[i]);
    }
  }

  if (success_poses.size() < 3)
    return false;

  cv::Mat R_cam2gripper, t_cam2gripper;
  cv::calibrateHandEye(Rs_gripper2base, ts_gripper2base, Rs_target2cam,
                       ts_target2cam, R_cam2gripper, t_cam2gripper,
                       cv::CALIB_HAND_EYE_PARK);

  // Cross-check against TSAI, a genuinely different algorithm (it solves
  // rotation from relative rotation *axes* between pose pairs, then
  // translation separately, vs PARK/HORAUD's simultaneous Kronecker-based
  // approach). When the samples carry enough independent information both
  // should land close together; a big gap here is itself the signal that
  // this sample set is too marginal to trust, regardless of which single
  // algorithm's number looks superficially "reasonable" -- PARK and HORAUD
  // agreeing with each other proves little since they're closely related
  // methods, not independent ones.
  cv::Mat R_cam2gripper_tsai, t_cam2gripper_tsai;
  cv::calibrateHandEye(Rs_gripper2base, ts_gripper2base, Rs_target2cam,
                       ts_target2cam, R_cam2gripper_tsai, t_cam2gripper_tsai,
                       cv::CALIB_HAND_EYE_TSAI);
  cross_check_discrepancy_mm_ =
      cv::norm(t_cam2gripper - t_cam2gripper_tsai);

  Eigen::Matrix3d R_x;
  Eigen::Vector3d t_x;
  cv::cv2eigen(R_cam2gripper, R_x);
  cv::cv2eigen(t_cam2gripper, t_x);

  // calibrateHandEye only returns the camera-to-gripper transform, not
  // "where the board is" -- unlike the old solver, which estimated that as
  // a fixed point directly. Reconstruct it the same way the per-point
  // error check already works: map every sample's own marker observation
  // through (this sample's robot pose) o (the shared hand-eye transform)
  // into the base/fixed frame; if the transform and the "board never
  // moved" assumption both hold, every sample should land on the same
  // point. Their mean *is* that estimated fixed point, and each sample's
  // distance from the mean is exactly the same per-point residual the
  // rest of this codebase (and calib.cpp's validation step) already
  // expects from getFixedPoint()/getErrors().
  std::vector<Eigen::Vector3d> p_base_per_sample;
  for (size_t k = 0; k < success_poses.size(); ++k) {
    Eigen::Matrix3d R_gb;
    Eigen::Vector3d t_gb, p_cam;
    cv::cv2eigen(Rs_gripper2base[k], R_gb);
    cv::cv2eigen(ts_gripper2base[k], t_gb);
    cv::cv2eigen(ts_target2cam[k], p_cam);
    Eigen::Vector3d p_gripper = R_x * p_cam + t_x;
    p_base_per_sample.push_back(R_gb * p_gripper + t_gb);
  }
  Eigen::Vector3d mean_p_base = Eigen::Vector3d::Zero();
  for (const auto &p : p_base_per_sample)
    mean_p_base += p;
  mean_p_base /= static_cast<double>(p_base_per_sample.size());

  errors_.assign(robot_poses_.size(), -1.0);
  size_t sample_idx = 0;
  for (size_t i = 0; i < robot_poses_.size(); ++i) {
    if (marker_success_[i] && sample_idx < p_base_per_sample.size()) {
      errors_[i] = (p_base_per_sample[sample_idx] - mean_p_base).norm();
      ++sample_idx;
    }
  }

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = R_x;
  transform.block<3, 1>(0, 3) = t_x;
  result_pose_ = Pose3D(Transform3D(transform));
  fixed_point_ = Vector3D(mean_p_base);
  return true;
}
