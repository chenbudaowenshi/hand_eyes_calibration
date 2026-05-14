#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief 3D 多点仿射拟合（思路类比 2D 四点插针：用 OpenCV 2D 版 estimateAffine2D，
 *        此处对 marker 与 robot 的三维坐标做最小二乘仿射标定）
 *
 * 模型：p_robot = A * p_marker + b，齐次形式 p_robot = M * [p_marker; 1]，M 为 3×4。
 * 至少需要 4 组非退化对应点；多于 4 点时做最小二乘。
 */
class MultiPoint3DCalibrator {
public:
  MultiPoint3DCalibrator() = default;
  virtual ~MultiPoint3DCalibrator() = default;

  /**
   * @param marker_points 相机/标记系下的三维点 (x,y,z)
   * @param robot_points  机器人系下对应三维点 (x,y,z)
   * @param transform_3x4  输出 CV_64FC1 的 3×4 矩阵 M
   * @return 成功返回 true
   */
  bool calibrate(const std::vector<cv::Point3d> &marker_points,
                 const std::vector<cv::Point3d> &robot_points,
                 cv::Mat &transform_3x4);

  cv::Point3d transform(const cv::Point3d &marker_point,
                        const cv::Mat &transform_3x4) const;

  double getRmse() const { return rmse_; }
  const std::vector<double> &getPerPointErrors() const { return per_point_errors_; }

private:
  double rmse_ = -1.0;
  std::vector<double> per_point_errors_;
};
