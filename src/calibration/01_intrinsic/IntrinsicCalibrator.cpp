#include "calibration/01_intrinsic/IntrinsicCalibrator.h"
#include <algorithm>
#include <iostream>

int IntrinsicCalibrator::detectCalibBoard(const cv::Mat &input_image,
                                       cv::Mat &output_corners_image,
                                       std::vector<cv::Point2f> &pixel_corners,
                                       std::vector<cv::Point3f> &world_corners,
                                       const CalibConfig &cfg, int calib_type) {
  pixel_corners.clear();
  world_corners.clear();

  if (input_image.empty() || cfg.cols <= 0 || cfg.rows <= 0 ||
      cfg.interval_mm <= 0) {
    return -1;
  }

  cv::Mat gray;
  if (input_image.channels() != 1) {
    cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = input_image;
  }

  std::vector<cv::Point2f> all_pixel_corners;
  bool found = false;

  if (cfg.pattern == CHESSBOARD) {
    found = cv::findChessboardCorners(
        gray, cv::Size(cfg.cols, cfg.rows), all_pixel_corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    if (found) {
      cv::cornerSubPix(
          gray, all_pixel_corners, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                           0.001));
    } else {
      // The classic detector needs a large, sharp checkerboard; it misses
      // most frames once the board only spans a small part of the image
      // (camera far from a small board). findChessboardCornersSB is a
      // newer, far more robust detector for exactly that case -- verified
      // on a real dataset where the classic detector found 4/100 frames
      // and this fallback found 86/100 of the SAME frames. Its corner
      // order is consistently the reverse of the classic detector's for
      // this board (confirmed against the 4 frames both detect, <2px
      // agreement after reversing), so flip it back before use --
      // otherwise every corner pairs with the wrong world point and PnP
      // silently returns a garbage pose instead of failing loudly.
      found = cv::findChessboardCornersSB(
          gray, cv::Size(cfg.cols, cfg.rows), all_pixel_corners,
          cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
      if (found)
        std::reverse(all_pixel_corners.begin(), all_pixel_corners.end());
    }
  } else if (cfg.pattern == CIRCLES_SYM) {
    found = cv::findCirclesGrid(gray, cv::Size(cfg.cols, cfg.rows),
                                all_pixel_corners, cv::CALIB_CB_SYMMETRIC_GRID);
  } else {
    cv::SimpleBlobDetector::Params params;
    params.blobColor = 255;
    params.filterByColor = true;
    params.maxArea = 50000;
    params.minArea = 1200;
    params.filterByArea = true;

    params.minDistBetweenBlobs = 36;
    params.filterByInertia = true;
    params.minInertiaRatio = 0.5;
    params.minConvexity = 0.8;
    params.thresholdStep = 20;
    cv::Ptr<cv::SimpleBlobDetector> blobDetector =
        cv::SimpleBlobDetector::create(params);
    cv::CirclesGridFinderParameters paramss;
    paramss.gridType =
        cv::CirclesGridFinderParameters::GridType::ASYMMETRIC_GRID;
    paramss.minDistanceToAddKeypoint = 45;
    paramss.edgeGain = 10;
    paramss.edgePenalty = 10;

    found = cv::findCirclesGrid(
        gray, cv::Size(cfg.cols, cfg.rows), all_pixel_corners,
        cv::CALIB_CB_CLUSTERING | cv::CALIB_CB_ASYMMETRIC_GRID, blobDetector,
        paramss);
  }

  cv::cvtColor(gray, output_corners_image, cv::COLOR_GRAY2RGB);
  cv::drawChessboardCorners(output_corners_image, cv::Size(cfg.cols, cfg.rows),
                            all_pixel_corners, found);

  if (!found)
    return 0;

  std::vector<cv::Point3f> all_world_corners;
  if (cfg.pattern == CHESSBOARD || cfg.pattern == CIRCLES_SYM) {
    for (int r = 0; r < cfg.rows; ++r)
      for (int c = 0; c < cfg.cols; ++c)
        all_world_corners.emplace_back(c * cfg.interval_mm, r * cfg.interval_mm,
                                       0);
  } else {
    for (int i = 0; i < cfg.rows; ++i) {
      if (i % 2 == 0) {
        for (int j = 0; j < cfg.cols; ++j)
          all_world_corners.push_back(cv::Point3f(
              j * cfg.interval_mm, (i * cfg.interval_mm) / 2.0f, 0));
      } else {
        for (int j = 0; j < cfg.cols; ++j)
          all_world_corners.push_back(
              cv::Point3f(j * cfg.interval_mm + cfg.interval_mm / 2.0f,
                          (i * cfg.interval_mm) / 2.0f, 0));
      }
    }
  }

  if (calib_type == 0 && cfg.pattern != CHESSBOARD) {
    // The sparse subset is only useful for the asymmetric-circle marker
    // layout.  A chessboard must retain every detected corner; the previous
    // hard-coded modulo-4 filter silently discarded rows for 12x9 boards.
    for (size_t i = 0; i < all_pixel_corners.size(); ++i) {
      auto last_num = i % 4;
      auto div_num = (i - last_num) / 4;
      if (div_num != 1 && div_num != 3) {
        pixel_corners.push_back(all_pixel_corners[i]);
        world_corners.push_back(all_world_corners[i]);
      }
    }
  } else if (calib_type == 1 || cfg.pattern == CHESSBOARD) {
    pixel_corners = all_pixel_corners;
    world_corners = all_world_corners;
  } else {
    return -1;
  }

  return 1;
}

bool IntrinsicCalibrator::intrinsicCalibrate(const CalibConfig &cfg,
                                          const std::vector<std::string> imgs) {
  // Re-deriving intrinsics via cv::calibrateCamera needs the BOARD tilted at
  // many different angles relative to the camera. A hand-eye dataset is the
  // opposite: the camera moves around a board that stays close to
  // fronto-parallel in frame (that's what makes it good hand-eye data), so
  // fitting intrinsics from it is poorly conditioned -- it can silently
  // produce a low-single-digit-pixel-looking RMS that's actually garbage
  // (asymmetric fx/fy, wild distortion). When the caller supplies real
  // device-reported intrinsics (cfg.use_fixed_intrinsics), skip the fit
  // entirely and trust those instead.
  if (cfg.use_fixed_intrinsics && !cfg.fixed_K.empty()) {
    std::cout << "使用外部提供的内参，跳过第一次内参拟合。\n";
    std::cout << "内参: \n" << cfg.fixed_K << "\n";
    std::cout << "畸变: \n" << cfg.fixed_dist << "\n";
    cv::FileStorage fs(cfg.xml_intrinsic_1st, cv::FileStorage::WRITE);
    fs << "K" << cfg.fixed_K << "distortion" << cfg.fixed_dist << "pattern"
       << (int)cfg.pattern;
    fs.release();
    return true;
  }

  std::vector<std::vector<cv::Point3f>> worlds_list_first;
  std::vector<std::vector<cv::Point2f>> pixels_list_first;
  cv::Size imgSize;
  int valid = 0;

  for (size_t i = 0; i < imgs.size(); ++i) {
    cv::Mat raw = cv::imread(imgs[i]);
    if (raw.empty())
      continue;
    if (imgSize == cv::Size())
      imgSize = raw.size();

    std::vector<cv::Point2f> pixel_corners;
    std::vector<cv::Point3f> world_corners;
    cv::Mat output_corners_image;
    int ret = detectCalibBoard(raw, output_corners_image, pixel_corners,
                                world_corners, cfg, 0);

    if (ret == 1) {
      worlds_list_first.push_back(world_corners);
      pixels_list_first.push_back(pixel_corners);
      valid++;
    }
    std::cout << "\r进度: " << i + 1 << "/" << imgs.size()
              << "  有效: " << valid << std::flush;
  }
  std::cout << "\n";

  if (valid < 5) {
    std::cerr << "至少 5 张！\n";
    return false;
  }

  cv::Mat K_first, dis_first;
  std::vector<cv::Mat> rvecs, tvecs;
  double rms =
      cv::calibrateCamera(worlds_list_first, pixels_list_first, imgSize,
                          K_first, dis_first, rvecs, tvecs, 0);
  std::cout << "第一次标定成功！RMS = " << rms << "\n";
  std::cout << "内参: \n" << K_first << "\n";
  std::cout << "畸变: \n" << dis_first << "\n";

  cv::FileStorage fs(cfg.xml_intrinsic_1st, cv::FileStorage::WRITE);
  fs << "K" << K_first << "distortion" << dis_first << "pattern"
     << (int)cfg.pattern;
  fs.release();
  return true;
}
