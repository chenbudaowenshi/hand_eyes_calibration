"""Small RealSense RGB-D capture wrapper used by the ROS publisher.

The module is intentionally import-side-effect free.  The camera must not be
opened at import time because ``run_realsense_viewer.py`` imports this class
before it creates its ROS node.
"""

import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:

    def __init__(self,
                 width=1280,
                 height=720,
                 fps=30,
                 timeout_ms=1000):

        self.timeout_ms = int(timeout_ms)
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than zero")

        self.pipeline = rs.pipeline()

        self.config = rs.config()

        self.config.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            fps)

        self.config.enable_stream(
            rs.stream.depth,
            width,
            height,
            rs.format.z16,
            fps)

        profile = self.pipeline.start(self.config)

        ##################################################
        # 深度对齐到彩色图
        ##################################################
        self.align = rs.align(rs.stream.color)

        ##################################################
        # 相机内参
        ##################################################
        color_profile = profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()

        intr = color_profile.get_intrinsics()

        self.fx = intr.fx
        self.fy = intr.fy
        self.cx = intr.ppx
        self.cy = intr.ppy

        self.width = intr.width
        self.height = intr.height

        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        ##################################################
        # 深度比例
        ##################################################
        depth_sensor = profile.get_device().first_depth_sensor()

        self.depth_scale = depth_sensor.get_depth_scale()

        print("=" * 60)
        print("RealSense Initialized")
        print("Resolution :", self.width, self.height)
        print("Depth Scale:", self.depth_scale)
        print("Camera Matrix:\n", self.K)
        print("=" * 60)

    ######################################################
    # 获取一帧
    ######################################################
    def get_frame(self):

        # Avoid blocking the ROS timer forever when USB streaming stalls.
        frames = self.pipeline.wait_for_frames(self.timeout_ms)

        frames = self.align.process(frames)

        color_frame = frames.get_color_frame()

        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None

        color = np.asanyarray(color_frame.get_data())

        depth = np.asanyarray(depth_frame.get_data())

        return {

            "color": color,

            "depth": depth,

            "K": self.K,

            "depth_scale": self.depth_scale,

            "timestamp": frames.get_timestamp()

        }

    ######################################################
    # 获取点云XYZ
    ######################################################
    def depth_to_xyz(self, depth):

        h, w = depth.shape

        u, v = np.meshgrid(np.arange(w), np.arange(h))

        z = depth.astype(np.float32) * self.depth_scale

        x = (u - self.cx) * z / self.fx

        y = (v - self.cy) * z / self.fy

        xyz = np.stack((x, y, z), axis=-1)

        return xyz

    ######################################################
    # 获取彩色深度图（方便调试）
    ######################################################
    @staticmethod
    def colorize_depth(depth):

        depth_vis = cv2.convertScaleAbs(
            depth,
            alpha=0.03)

        depth_vis = cv2.applyColorMap(
            depth_vis,
            cv2.COLORMAP_JET)

        return depth_vis

    ######################################################
    # 关闭相机
    ######################################################
    def stop(self):

        self.pipeline.stop()


def _preview() -> None:
    """Display a local RGB/depth preview when this file is run directly."""

    camera = RealSenseCamera()
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            color = frame["color"]
            depth_vis = camera.colorize_depth(frame["depth"])
            cv2.imshow("RGB | Depth", cv2.hconcat([color, depth_vis]))
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _preview()
