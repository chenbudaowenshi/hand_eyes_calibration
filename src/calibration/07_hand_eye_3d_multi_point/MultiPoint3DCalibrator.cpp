#include "calibration/07_hand_eye_3d_multi_point/MultiPoint3DCalibrator.h"

#include <cmath>

#include <Eigen/Dense>

bool MultiPoint3DCalibrator::calibrate(const std::vector<cv::Point3d> &marker_points,
                                     const std::vector<cv::Point3d> &robot_points,
                                     cv::Mat &transform_3x4) {
  rmse_ = -1.0;
  per_point_errors_.clear();

  const size_t n = marker_points.size();
  if (n < 4 || robot_points.size() != n) {
    transform_3x4.release();
    return false;
  }

  Eigen::MatrixXd P(static_cast<Eigen::Index>(n), 4);
  Eigen::MatrixXd Q(static_cast<Eigen::Index>(n), 3);
  for (size_t i = 0; i < n; ++i) {
    P(static_cast<Eigen::Index>(i), 0) = marker_points[i].x;
    P(static_cast<Eigen::Index>(i), 1) = marker_points[i].y;
    P(static_cast<Eigen::Index>(i), 2) = marker_points[i].z;
    P(static_cast<Eigen::Index>(i), 3) = 1.0;
    Q(static_cast<Eigen::Index>(i), 0) = robot_points[i].x;
    Q(static_cast<Eigen::Index>(i), 1) = robot_points[i].y;
    Q(static_cast<Eigen::Index>(i), 2) = robot_points[i].z;
  }

  Eigen::MatrixXd MT = P.colPivHouseholderQr().solve(Q);
  if (!MT.allFinite()) {
    transform_3x4.release();
    return false;
  }

  Eigen::MatrixXd M = MT.transpose();
  transform_3x4 = cv::Mat(3, 4, CV_64F);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      transform_3x4.at<double>(r, c) = M(r, c);
    }
  }

  double sum_sq = 0.0;
  per_point_errors_.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    cv::Point3d pred = transform(marker_points[i], transform_3x4);
    const double ex = pred.x - robot_points[i].x;
    const double ey = pred.y - robot_points[i].y;
    const double ez = pred.z - robot_points[i].z;
    const double err = std::sqrt(ex * ex + ey * ey + ez * ez);
    per_point_errors_.push_back(err);
    sum_sq += ex * ex + ey * ey + ez * ez;
  }
  rmse_ = std::sqrt(sum_sq / static_cast<double>(n));
  return true;
}

cv::Point3d MultiPoint3DCalibrator::transform(const cv::Point3d &marker_point,
                                             const cv::Mat &transform_3x4) const {
  if (transform_3x4.empty() || transform_3x4.rows != 3 ||
      transform_3x4.cols != 4 || transform_3x4.type() != CV_64F) {
    return cv::Point3d(0.0, 0.0, 0.0);
  }
  const double x = marker_point.x;
  const double y = marker_point.y;
  const double z = marker_point.z;
  cv::Point3d out;
  out.x = transform_3x4.at<double>(0, 0) * x + transform_3x4.at<double>(0, 1) * y +
          transform_3x4.at<double>(0, 2) * z + transform_3x4.at<double>(0, 3);
  out.y = transform_3x4.at<double>(1, 0) * x + transform_3x4.at<double>(1, 1) * y +
          transform_3x4.at<double>(1, 2) * z + transform_3x4.at<double>(1, 3);
  out.z = transform_3x4.at<double>(2, 0) * x + transform_3x4.at<double>(2, 1) * y +
          transform_3x4.at<double>(2, 2) * z + transform_3x4.at<double>(2, 3);
  return out;
}
