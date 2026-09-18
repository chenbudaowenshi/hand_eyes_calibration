#include "calibration/hand_eye_3d_base/RobotCameraCalibrator3D.h"

#include <Eigen/LU>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>

#define PI 3.1415926

void RobotCameraCalibrator3DImpl::inputData(
    std::vector<Eigen::Vector3d> &_markPoint,
    std::vector<Eigen::Matrix<double, 4, 4, Eigen::DontAlign>> &_robotPose,
    CalibrationType3D type) {
  Eigen::Matrix<double, 3, 5> p_p;
  for (int i = 0; i < _markPoint.size(); i++) {
    p_p.block<3, 4>(0, 0) = _robotPose[i].block<3, 4>(0, 0);
    p_p(0, 4) = _markPoint[i](0);
    p_p(1, 4) = _markPoint[i](1);
    p_p(2, 4) = _markPoint[i](2);

    if (type == EIH) {
      p_p.topLeftCorner<3, 3>() = p_p.topLeftCorner<3, 3>().transpose().eval();
      p_p.middleCols<1>(3) = -p_p.topLeftCorner<3, 3>() * p_p.middleCols<1>(3);
    }
    Pose_Point.push_back(p_p);
  }
}

Eigen::Matrix3d RobotCameraCalibrator3DImpl::skew(Eigen::Vector3d const &axis) {
  Eigen::Matrix3d out;
  out(0, 0) = 0;
  out(0, 1) = -axis(2);
  out(0, 2) = axis(1);
  out(1, 0) = axis(2);
  out(1, 1) = 0;
  out(1, 2) = -axis(0);
  out(2, 0) = -axis(1);
  out(2, 1) = axis(0);
  out(2, 2) = 0;

  return out;
}

Eigen::Matrix<double, 9, 1> RobotCameraCalibrator3DImpl::update() {
  int n = Pose_Point.size();

  Eigen::Matrix3d R_BW = Eye2Fixed.leftCols<3>();
  Eigen::AngleAxisd axis(R_BW);

  double axis_norm = axis.angle();

  Eigen::Matrix3d axis_skew;
  axis_skew = skew(axis_norm * axis.axis());

  Eigen::Matrix3d Q_BW =
      R_BW.transpose() *
      (Eigen::MatrixXd::Identity(3, 3) +
       1 / axis_norm / axis_norm * (1 - cos(axis_norm)) * axis_skew +
       1 / axis_norm / axis_norm / axis_norm * (axis_norm - sin(axis_norm)) *
           axis_skew * axis_skew);

  Eigen::Matrix<double, Eigen::Dynamic, 3> J1;
  Eigen::Matrix<double, Eigen::Dynamic, 3> J2;
  Eigen::Matrix<double, Eigen::Dynamic, 3> J3;
  Eigen::Matrix<double, Eigen::Dynamic, 9> J;
  Eigen::Matrix<double, Eigen::Dynamic, 1> f;

  J1.resize(3 * n, 3);
  J2.resize(3 * n, 3);
  J3.resize(3 * n, 3);
  J.resize(3 * n, 9);
  f.resize(3 * n, 1);

  Eigen::Matrix3d crossRP_W;
  for (int i = 1; i <= n; i++) {
    crossRP_W = -R_BW * skew(Pose_Point[i - 1].rightCols<1>()) * Q_BW;
    J1.block<3, 3>(3 * i - 3, 0) = crossRP_W;

    J3.block<3, 3>(3 * i - 3, 0) = Pose_Point[i - 1].leftCols<3>();

    f.block<3, 1>(3 * i - 3, 0) =
        -R_BW * Pose_Point[i - 1].rightCols<1>() - Eye2Fixed.rightCols<1>() +
        Pose_Point[i - 1].leftCols<3>() * PointInFixed +
        Pose_Point[i - 1].middleCols<1>(3);
  }

  J2 = Eigen::MatrixXd::Identity(3, 3).replicate(n, 1);
  J << J1, J2, J3;

  // SVD-based solve instead of (J^T J).inverse() * J^T * f: the normal-
  // equations form squares J's condition number, so on marginal/ill-
  // conditioned data (few samples, or rotation not exciting all axes) the
  // plain inverse amplifies noise into a huge, wrong-but-plausible-looking
  // update instead of the bounded, minimum-norm least-squares correction
  // an SVD solve gives. J has a compile-time-fixed column count (9), and
  // Eigen's thin-SVD path requires a dynamic column count, so use full
  // U/V here -- J is small (3n x 9) so the extra cost is negligible.
  Eigen::Matrix<double, 9, 1> x =
      J.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(f);

  return x;
}

int RobotCameraCalibrator3DImpl::calib_closedForm() {
  int posNum = Pose_Point.size();

  Eigen::Matrix<double, Eigen::Dynamic, 15> K;

  Eigen::Matrix<double, Eigen::Dynamic, 1> b;

  Eigen::Matrix<double, 15, 1> x;

  K.resize(3 * posNum, 15);
  b.resize(3 * posNum, 1);
  for (int i = 1; i <= posNum; i++) {
    K.middleRows<3>(i * 3 - 3)
        << Eigen::kroneckerProduct(Pose_Point[i - 1].rightCols<1>().transpose(),
                                   Eigen::MatrixXd::Identity(3, 3)),
        Eigen::MatrixXd::Identity(3, 3), -Pose_Point[i - 1].leftCols<3>();
    b.middleRows<3>(i * 3 - 3) << Pose_Point[i - 1].middleCols<1>(3);
  }

  // Same reasoning as update(): solve K directly via SVD instead of
  // inverting the normal-equations matrix (K^T K), which squares K's
  // condition number. A 2-DOF mechanism (e.g. a pan/tilt head with no
  // roll) can only excite a subset of this system's 15 unknowns well no
  // matter how the samples are spread out -- when that happens, a plain
  // inverse blows the near-singular directions up into a wrong answer
  // that still "looks like" a solution (no error, no NaN); SVD gracefully
  // returns the minimum-norm least-squares solution instead. K has a
  // compile-time-fixed column count (15), and Eigen's thin-SVD path
  // requires a dynamic column count, so use full U/V here -- K is small
  // (3n x 15) so the extra cost is negligible.
  Eigen::BDCSVD<Eigen::MatrixXd> solve_svd(
      K, Eigen::ComputeFullU | Eigen::ComputeFullV);
  x = solve_svd.solve(b);
  {
    const auto &sv = solve_svd.singularValues();
    double cond = sv(0) / sv(sv.size() - 1);
    closedFormConditionNumber = cond;
    if (cond > 1e4) {
      std::cerr << "\033[1;33m[WARN] Hand-eye linear system is poorly "
                   "conditioned (condition number "
                << cond
                << "). The samples likely don't excite enough independent "
                   "rotation directions (e.g. a pan/tilt mechanism with no "
                   "roll, or poses clustered too close together) -- the "
                   "result below is the minimum-norm least-squares answer, "
                   "not a well-determined one. Treat it with suspicion.\033[0m"
                << std::endl;
    }
  }

  Eigen::Matrix<double, 3, 3> temp_x;
  temp_x = Eigen::Map<Eigen::Matrix<double, 3, 3>, Eigen::RowMajor>(
      x.topRows<9>().data(), 3, 3);

  Eye2Fixed.rightCols<1>() << x.middleRows<3>(9);
  PointInFixed = x.bottomRows<3>();
  Eigen::Matrix<double, 3, 3> U;
  Eigen::Matrix<double, 3, 3> V;

  Eigen::BDCSVD<Eigen::MatrixXd> svd(temp_x,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
  U = svd.matrixU().block(0, 0, 3, 3);
  V = svd.matrixV().block(0, 0, 3, 3);
  Eye2Fixed.leftCols<3>() = U * V.transpose();

  return 0;
}

int RobotCameraCalibrator3DImpl::calib_iterative() {
  Eigen::Matrix<double, 9, 1> x = Eigen::MatrixXd::Ones(9, 1);

  int count = 0;
  while (x.norm() > 1e-10 && count < 100) {
    x = update();
    Eigen::Matrix3d R_BW = Eye2Fixed.leftCols<3>();
    Eigen::AngleAxisd axis(R_BW);
    Eigen::Vector3d new_AngleAxis = axis.angle() * axis.axis() + x.topRows<3>();
    R_BW = Eigen::AngleAxisd(new_AngleAxis.norm(),
                             new_AngleAxis / new_AngleAxis.norm());
    Eye2Fixed.leftCols<3>() = R_BW;
    Eye2Fixed.rightCols<1>() = Eye2Fixed.rightCols<1>() + x.middleRows<3>(3);
    PointInFixed = PointInFixed - x.bottomRows<3>();
    count++;
  }
  return 0;
}

double RobotCameraCalibrator3DImpl::calibrate() {
  if (Pose_Point.size() < 5)
    return -1;

  calib_closedForm();
  Eigen::Vector3d error;
  double rmse = 0.0;
  // Always refine, even when calib_closedForm() flagged a high condition
  // number: that number measures worst-case noise sensitivity of the
  // *linear* closed-form system, not whether Gauss-Newton refinement will
  // help. In practice it does help on real, merely-imperfect datasets
  // (skipping it here regressed a known-good 30-pose run's fit from
  // ~25mm RMS to >1000mm) -- the condition-number warning stays purely
  // informational; getCalibrationError()'s per-point residuals and
  // calib.cpp's plausibility check on the final translation are what
  // actually catch a bad answer.
  calib_iterative();
  for (int i = 1; i <= Pose_Point.size(); i++) {
    error = -Eye2Fixed.leftCols<3>() * Pose_Point[i - 1].rightCols<1>() -
            Eye2Fixed.rightCols<1>() +
            Pose_Point[i - 1].leftCols<3>() * PointInFixed +
            Pose_Point[i - 1].middleCols<1>(3);
    rmse += error.norm();
  }
  errorValue = sqrt(rmse / Pose_Point.size());
  return errorValue;
}

RobotCameraCalibrator3D::RobotCameraCalibrator3D()
    : pImpl(new RobotCameraCalibrator3DImpl) {}

double RobotCameraCalibrator3D::calibrate(
    std::vector<Eigen::Vector3d> &_markPoint,
    std::vector<Eigen::Matrix<double, 4, 4, Eigen::DontAlign>> &_robotPose,
    CalibrationType3D type, Eigen::Matrix4d &_Transform,
    Eigen::Vector3d &_TCP) {
  if ((_markPoint.size() != _robotPose.size()) || _robotPose.size() < 5) {
    std::cout << _markPoint.size() << "," << _robotPose.size();
    throw nullptr;
  }
  pImpl->inputData(_markPoint, _robotPose, type);
  double out = pImpl->calibrate();
  _Transform = Eigen::Matrix4d::Identity();
  _Transform.block<3, 4>(0, 0) = pImpl->Eye2Fixed;
  _TCP = pImpl->PointInFixed;

  calibration_error_.clear();
  for (size_t i = 0; i < _markPoint.size(); ++i) {
    Eigen::Vector3d error;
    if (type == CalibrationType3D::ETH) {
      error = -_Transform.block<3, 3>(0, 0) * _markPoint[i] -
              _Transform.block<3, 1>(0, 3) +
              _robotPose[i].block<3, 3>(0, 0) * _TCP +
              _robotPose[i].block<3, 1>(0, 3);
    } else {
      error = _TCP - (_robotPose[i].block<3, 3>(0, 0) *
                          (_Transform.block<3, 3>(0, 0) * _markPoint[i] +
                           _Transform.block<3, 1>(0, 3)) +
                      _robotPose[i].block<3, 1>(0, 3));
    }
    calibration_error_.push_back(error.norm());
  }

  return out;
}

double RobotCameraCalibrator3D::calibrate(
    std::vector<Eigen::Vector3d> &_markPoint, std::vector<Pose3D> &_robotPose,
    CalibrationType3D type, Eigen::Matrix4d &_Transform,
    Eigen::Vector3d &_TCP) {

  std::vector<Eigen::Matrix<double, 4, 4, Eigen::DontAlign>> robot_poses_mat;
  for (auto &p : _robotPose) {
    Pose3D p_converted = p;
    if (rpy_type_ != RPY::RPYType::XYZ ||
        ref_type_ != RPY::ReferenceType::EXTRINSIC) {
      RPY correct_rpy(p.rx(), p.ry(), p.rz(), rpy_type_, ref_type_);
      p_converted = Pose3D(p.xyz(), correct_rpy);
    }
    robot_poses_mat.push_back(p_converted.toTransform3D().eigen());
  }

  return calibrate(_markPoint, robot_poses_mat, type, _Transform, _TCP);
}

RobotCameraCalibrator3D::~RobotCameraCalibrator3D() {
  if (pImpl)
    delete pImpl;
  pImpl = nullptr;
}
