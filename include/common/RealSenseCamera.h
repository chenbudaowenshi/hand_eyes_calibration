#ifndef REALSENSECAMERA_H
#define REALSENSECAMERA_H

#include <chrono>               // Potential delay in main loop
#include <iostream>             // Console output
#include <librealsense2/rs.hpp> // RealSense SDK header
#include <mutex>                // Thread-safe shared data
#include <opencv2/opencv.hpp>   // OpenCV header
#include <thread>               // Sleep in main loop
#include <vector>

// RealSense 相机内参/深度尺度等元数据，供标定数据采集流程记录使用
struct RealSenseCameraInfo {
  cv::Mat K;              // 3x3 彩色相机内参矩阵
  cv::Mat dist;           // 1x5 彩色相机畸变系数
  float depth_scale = 0;  // 深度单位换算比例 (米/LSB)
  int width = 0;          // 彩色流宽度
  int height = 0;         // 彩色流高度
  int fps = 0;            // 彩色流帧率
};

// RealSenseCamera 类，用于初始化和帧回调
class RealSenseCamera {
private:
  rs2::pipeline pipe;           // RealSense 管道
  rs2::align align_to_color;    // 颜色对齐器
  cv::Mat latest_color;         // 最新颜色图
  cv::Mat latest_depth_aligned; // 最新对齐后的深度图
  double latest_color_timestamp_ms_ = 0.0; // RealSense device timestamp (ms)
  double latest_capture_stamp_s_ = 0.0; // host system-clock capture time (Unix s)
  std::mutex data_mutex;        // 数据互斥锁
  bool has_new_frame = false;   // 是否有新帧
  int frame_count = 0;          // 帧计数
  bool is_started_ = false;     // 管道是否成功启动 (start() 抛异常时为 false)

  // 当新帧到达时触发的回调函数
  void frame_callback(const rs2::frame &frame) {
    if (auto frameset = frame.as<rs2::frameset>()) {
      frame_count++;
      if (frame_count <= 20)
        return;

      auto aligned = align_to_color.process(frameset);
      const auto capture_now = std::chrono::system_clock::now();
      const double capture_stamp_s =
          std::chrono::duration<double>(capture_now.time_since_epoch()).count();

      if (auto color = aligned.get_color_frame()) {
        cv::Mat img(cv::Size(color.get_width(), color.get_height()), CV_8UC3,
                    (void *)color.get_data(), cv::Mat::AUTO_STEP);
        std::lock_guard<std::mutex> lock(data_mutex);
        latest_color = img.clone();
        // rs2 时间戳单位为毫秒；无硬件时间同步时默认取自主机系统时钟
        // (RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME)，与 ROS2 系统时钟同源可比对。
        latest_color_timestamp_ms_ = color.get_timestamp();
        latest_capture_stamp_s_ = capture_stamp_s;
        has_new_frame = true;
      }

      if (auto depth = aligned.get_depth_frame()) {
        cv::Mat dep(cv::Size(depth.get_width(), depth.get_height()), CV_16U,
                    (void *)depth.get_data(), cv::Mat::AUTO_STEP);
        std::lock_guard<std::mutex> lock(data_mutex);
        latest_depth_aligned = dep.clone();
        has_new_frame = true;
      }
    }
  }

public:
  // 构造函数：初始化管道
  RealSenseCamera() : align_to_color(RS2_STREAM_COLOR) {}
  ~RealSenseCamera() { stop(); }

  /**
   * @brief 启动相机流
   * @param width [in] 图像宽度
   * @param height [in] 图像高度
   * @param fps [in] 帧率
   * @return 启动成功返回 true；设备未插上/请求的分辨率帧率不受支持等情况返回
   *         false，调用方必须检查这个返回值 -- 否则后续等待帧的循环会永远等
   *         不到数据而卡死（管道根本没起来）。
   */
  bool start(int width, int height, int fps) {
    try {
      rs2::config cfg;
      // 彩色 (RGB) 传感器与深度 (stereo) 传感器的原生分辨率集合并不相同 --
      // 例如 D435i 彩色最高支持 1920x1080, 但深度传感器不支持该分辨率.
      // 两路用同一个 width/height 请求会导致深度流无法解析, 进而使整个
      // pipeline.start() 失败 (表现为 "Couldn't resolve requests", 容易被
      // 误判成"相机没插好"). 深度流固定用一个 D400 系列几乎全系支持的安全
      // 分辨率, 与彩色分辨率解耦; rs2::align 会自动把深度重投影到彩色视口.
      constexpr int kSafeDepthWidth = 640;
      constexpr int kSafeDepthHeight = 480;
      // 配置流（颜色和深度）
      cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8,
                        fps); // 颜色流 (BGR8)
      cfg.enable_stream(RS2_STREAM_DEPTH, kSafeDepthWidth, kSafeDepthHeight,
                        RS2_FORMAT_Z16, fps); // 深度流 (可选)

      // 使用回调 lambda 启动管道
      pipe.start(cfg, [this](const rs2::frame &frame) {
        this->frame_callback(frame);
      });
      std::cout << "RealSense 相机已启动。" << std::endl;
      is_started_ = true;
      return true;
    } catch (const rs2::error &e) {
      std::cerr << "RealSense 错误调用 " << e.get_failed_function() << "("
                << e.get_failed_args() << "):\n    " << e.what() << std::endl;
      return false;
    } catch (const std::exception &e) {
      std::cerr << "异常: " << e.what() << std::endl;
      return false;
    }
  }

  /**
   * @brief 停止相机流
   */
  void stop() {
    // Guard against a pipeline that never actually started (start() threw
    // and was caught): pipe.stop() on an unstarted pipe would also throw.
    if (!is_started_)
      return;
    // start() runs the pipeline in callback mode (frame_callback above), and
    // librealsense explicitly documents that poll_for_frames()/
    // wait_for_frames() throw once a pipeline was started with a callback --
    // do not call them here. Just stop; frame_callback stops firing once
    // pipe.stop() returns.
    pipe.stop();
    std::cout << "RealSense 相机已停止。" << std::endl;
    is_started_ = false;
  }

  /**
   * @brief 获取最新颜色图
   * @return OpenCV 矩阵格式的颜色图
   */
  cv::Mat getLatestColor() {
    std::lock_guard<std::mutex> lock(data_mutex);
    has_new_frame = false;
    return latest_color.clone();
  }

  /**
   * @brief 获取最新深度图
   * @return OpenCV 矩阵格式的深度图
   */
  cv::Mat getLatestDepth() {
    std::lock_guard<std::mutex> lock(data_mutex);
    has_new_frame = false;
    return latest_depth_aligned.clone();
  }

  /**
   * @brief 检查是否有新帧
   * @return 如果有新帧返回 true，否则返回 false
   */
  bool hasNewFrame() {
    std::lock_guard<std::mutex> lock(data_mutex);
    return has_new_frame;
  }

  /**
   * @brief 获取最新彩色帧的采集时间戳
   * @return 采集时间戳 (毫秒, RealSense 设备时钟域)
   */
  double getLatestColorTimestampMs() {
    std::lock_guard<std::mutex> lock(data_mutex);
    return latest_color_timestamp_ms_;
  }

  /**
   * @brief Return one color frame and its matching host capture timestamp.
   * @param color [out] copied BGR frame
   * @param stamp_s [out] Unix/system-clock seconds captured in the callback
   * @return false when no frame has been received
   */
  bool getLatestColorWithTimestamp(cv::Mat &color, double &stamp_s) {
    std::lock_guard<std::mutex> lock(data_mutex);
    if (latest_color.empty() || latest_capture_stamp_s_ <= 0.0)
      return false;
    color = latest_color.clone();
    stamp_s = latest_capture_stamp_s_;
    has_new_frame = false;
    return true;
  }

  /**
   * @brief 打印相机内参信息
   */
  void printIntrinsics() {
    try {
      auto profile = pipe.get_active_profile();
      auto color_stream =
          profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
      auto depth_stream =
          profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();

      auto color_intrinsics = color_stream.get_intrinsics();
      auto depth_intrinsics = depth_stream.get_intrinsics();

      std::cout << "颜色相机内参:" << std::endl;
      std::cout << "  宽度: " << color_intrinsics.width
                << ", 高度: " << color_intrinsics.height << std::endl;
      std::cout << "  fx: " << color_intrinsics.fx
                << ", fy: " << color_intrinsics.fy << std::endl;
      std::cout << "  ppx: " << color_intrinsics.ppx
                << ", ppy: " << color_intrinsics.ppy << std::endl;
      std::cout << "  畸变模型: " << color_intrinsics.model << ", 畸变系数: ";
      for (int i = 0; i < 5; ++i) {
        std::cout << color_intrinsics.coeffs[i] << " ";
      }
      std::cout << std::endl;

      std::cout << "深度相机内参:" << std::endl;
      std::cout << "  宽度: " << depth_intrinsics.width
                << ", 高度: " << depth_intrinsics.height << std::endl;
      std::cout << "  fx: " << depth_intrinsics.fx
                << ", fy: " << depth_intrinsics.fy << std::endl;
      std::cout << "  ppx: " << depth_intrinsics.ppx
                << ", ppy: " << depth_intrinsics.ppy << std::endl;
      std::cout << "  畸变模型: " << depth_intrinsics.model << ", 畸变系数: ";
      for (int i = 0; i < 5; ++i) {
        std::cout << depth_intrinsics.coeffs[i] << " ";
      }
      std::cout << std::endl;
    } catch (const rs2::error &e) {
      std::cerr << "获取内参时出错: " << e.what() << std::endl;
    }
  }

  /**
   * @brief 获取颜色相机内参矩阵
   * @return 3x3 相机内参矩阵
   */
  cv::Mat getColorCameraMatrix() const {
    try {
      auto profile = pipe.get_active_profile();
      auto stream =
          profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
      auto intr = stream.get_intrinsics();

      cv::Mat K = (cv::Mat_<double>(3, 3) << intr.fx, 0, intr.ppx, 0, intr.fy,
                   intr.ppy, 0, 0, 1);
      return K;
    } catch (...) {
      std::cerr << "警告：获取颜色内参失败，使用默认值！" << std::endl;
      return (cv::Mat_<double>(3, 3) << 608, 0, 320, 0, 608, 240, 0, 0, 1);
    }
  }

  /**
   * @brief 获取颜色相机畸变系数
   * @return 1x5 畸变系数矩阵
   */
  cv::Mat getColorDistCoeffs() const {
    try {
      auto profile = pipe.get_active_profile();
      auto stream =
          profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
      auto intr = stream.get_intrinsics();

      cv::Mat dist = cv::Mat(1, 5, CV_64F);
      for (int i = 0; i < 5; ++i) {
        dist.at<double>(0, i) = intr.coeffs[i];
      }
      return dist;
    } catch (...) {
      std::cerr << "警告：获取畸变系数失败，返回全 0！" << std::endl;
      return cv::Mat::zeros(1, 5, CV_64F);
    }
  }

  /**
   * @brief 获取设备真实的内参、畸变、深度尺度、分辨率与帧率
   * @return 相机元数据 (用于标定数据集留档，与二次棋盘格标定结果分开保存)
   */
  RealSenseCameraInfo getCameraInfo() const {
    RealSenseCameraInfo info;
    info.K = getColorCameraMatrix();
    info.dist = getColorDistCoeffs();
    try {
      auto profile = pipe.get_active_profile();
      auto color_stream =
          profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
      info.width = color_stream.width();
      info.height = color_stream.height();
      info.fps = color_stream.fps();

      for (auto &sensor : profile.get_device().query_sensors()) {
        if (sensor.is<rs2::depth_sensor>()) {
          info.depth_scale = sensor.as<rs2::depth_sensor>().get_depth_scale();
          break;
        }
      }
    } catch (const rs2::error &e) {
      std::cerr << "警告：获取相机元数据失败：" << e.what() << std::endl;
    }
    return info;
  }

  std::vector<double> calibration_error_; // 标定误差
};

#endif // REALSENSECAMERA_H
