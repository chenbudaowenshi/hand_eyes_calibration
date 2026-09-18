#pragma once

#include "calibration/01_intrinsic/IntrinsicCalibrator.h"
#include "calibration/hand_eye_3d_base/RobotCameraCalibrator3D.h"
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief 基于标定板图像的 3D 手眼标定类
 */
class BoardImageCalibrator {
public:
  BoardImageCalibrator() {}
  virtual ~BoardImageCalibrator() {}

  /**
   * @brief 设置标定配置
   */
  void setConfig(const CalibConfig &cfg) { cfg_ = cfg; }

  /**
   * @brief 设置图像文件列表
   */
  void setImageFiles(const std::vector<std::string> &imgs) {
    img_files_ = imgs;
  }

  /**
   * @brief 设置机器人位姿列表
   */
  void setRobotPoses(const std::vector<Pose3D> &poses) { robot_poses_ = poses; }

  /**
   * @brief 设置标定类型
   */
  void setCalibrationType(CalibrationType3D type) { calib_type_ = type; }

  /**
   * @brief 执行相机外参标定 (手眼标定)
   */
  Pose3D extrinsicCalibrate(const CalibConfig &cfg,
                            const std::vector<std::string> imgs,
                            const std::vector<Pose3D> &robot_poses,
                            CalibrationType3D calib_type);

  /**
   * @brief 执行标定
   * @return 成功返回 true
   */
  virtual bool runCalibration();

  /**
   * @brief 获取标定结果
   */
  Pose3D getResultPose() const { return result_pose_; }

  /**
   * @brief 获取标定误差列表
   */
  std::vector<double> getErrors() const { return errors_; }

  /**
   * @brief 获取标记点列表 (相机坐标系)
   */
  std::vector<Vector3D> getMarkerPoints() const { return marker_points_; }

  /**
   * @brief 获取每组数据是否成功检测到标记点
   */
  std::vector<bool> getMarkerSuccess() const { return marker_success_; }

  /**
   * @brief 获取标定板/标记点在固定坐标系下的位置
   * EIH: 固定坐标系为机器人基座系; ETH: 固定坐标系为末端法兰系.
   */
  Vector3D getFixedPoint() const { return fixed_point_; }

  /**
   * @brief 获取二次标定 (基于全部有效图像) 得到的内参与畸变
   */
  void getRefinedIntrinsics(cv::Mat &K, cv::Mat &dist) const {
    K = refined_K_;
    dist = refined_dist_;
  }

  /**
   * @brief 用另一种独立的经典算法 (TSAI) 交叉核验主结果 (PARK) 的平移量,
   * 两者相差越大, 说明样本本身信息量不足以稳定确定解, 不是某一种算法的实现
   * 问题. 单位 mm.
   */
  double getCrossCheckTranslationDiscrepancyMm() const {
    return cross_check_discrepancy_mm_;
  }

  /**
   * @brief 设置欧拉角类型
   */
  void setRPYType(RPY::RPYType type,
                  RPY::ReferenceType ref = RPY::ReferenceType::EXTRINSIC) {
    rpy_type_ = type;
    ref_type_ = ref;
  }

protected:
  CalibConfig cfg_;
  std::vector<std::string> img_files_;
  std::vector<Pose3D> robot_poses_;
  CalibrationType3D calib_type_ = CalibrationType3D::ETH;
  Pose3D result_pose_;
  std::vector<double> errors_;
  std::vector<Vector3D> marker_points_; // 存储每个位姿对应的标记点
  std::vector<bool> marker_success_;    // 存储每个位姿是否检测成功
  Vector3D fixed_point_;                // 标记点在固定坐标系下的位置
  cv::Mat refined_K_;                   // 二次标定内参
  cv::Mat refined_dist_;                // 二次标定畸变
  double cross_check_discrepancy_mm_ = -1; // PARK 与 TSAI 平移量之差

  RPY::RPYType rpy_type_ = RPY::RPYType::XYZ;
  RPY::ReferenceType ref_type_ = RPY::ReferenceType::EXTRINSIC;
};
