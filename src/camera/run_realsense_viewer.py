#!/usr/bin/env python3
"""Publish aligned RealSense RGB-D frames (and optional IMU) over ROS 2."""

from __future__ import annotations

import argparse
import fcntl
import inspect
import os
import sys
import time
import traceback
from typing import Any

import numpy as np


LOCKFILE_PATH = "/tmp/realsense_viewer.lock"
_lock_file = None


ROS_IMPORT_ERROR: Exception | None = None
try:
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import CameraInfo, Image, Imu
    from std_msgs.msg import Float32, String
except Exception as exc:  # pragma: no cover - depends on the ROS installation
    ROS_IMPORT_ERROR = exc
    rclpy = None
    Node = object
    TransformStamped = None
    CameraInfo = None
    Image = None
    Imu = None
    Float32 = None
    String = None
    QoSProfile = None
    QoSHistoryPolicy = None
    QoSReliabilityPolicy = None
    QoSDurabilityPolicy = None

try:  # Static TF is optional and disabled by default.
    import tf2_ros
except Exception:  # pragma: no cover - depends on the ROS installation
    tf2_ros = None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite value greater than zero")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a RealSense RGB-D stream as ROS 2 sensor topics."
    )
    parser.add_argument("--width", type=positive_int, default=1280)
    parser.add_argument("--height", type=positive_int, default=720)
    parser.add_argument(
        "--camera-fps",
        "--fps",
        dest="camera_fps",
        type=positive_int,
        default=30,
        help="RealSense RGB/depth capture rate (default: 30).",
    )
    parser.add_argument(
        "--publish-fps",
        type=positive_float,
        default=5.0,
        help="ROS RGB-D publish rate. Keep this modest for Wi-Fi (default: 5).",
    )
    parser.add_argument(
        "--use-imu",
        action="store_true",
        help="Enable and publish the D435i IMU. Disabled by default.",
    )
    parser.add_argument("--accel-fps", type=positive_int, default=100)
    parser.add_argument("--gyro-fps", type=positive_int, default=200)
    parser.add_argument("--frame-id", default="camera_link")
    parser.add_argument(
        "--imu-frame-id",
        default=None,
        help="Imu.header.frame_id (default: the value of --frame-id).",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=positive_int,
        default=25,
        help="Exit nonzero after this many consecutive capture/publish errors.",
    )
    parser.add_argument(
        "--publish-static-tf",
        action="store_true",
        help=(
            "Publish an identity static transform from --static-parent-frame to "
            "--frame-id. Disabled by default; use calibrated robot TF for IK."
        ),
    )
    parser.add_argument("--static-parent-frame", default="world")
    parser.add_argument(
        "--opencv-only",
        action="store_true",
        help="Run a local OpenCV preview without creating a ROS node.",
    )
    return parser


def check_realsense_imu_permission() -> None:
    for device_name in ("iio:device0", "iio:device1"):
        scan_path = os.path.join("/sys/bus/iio/devices", device_name, "scan_elements")
        if not os.path.isdir(scan_path):
            continue
        print(f"[INFO] Found IMU: {scan_path}", flush=True)
        test_file = os.path.join(scan_path, "in_anglvel_x_en")
        if os.access(test_file, os.W_OK):
            print("[INFO] IMU sysfs permission OK", flush=True)
        else:
            print(
                "[WARN] IMU sysfs permission denied; librealsense may still work "
                "through its installed udev rules.",
                flush=True,
            )
        return
    print("[WARN] No RealSense IMU sysfs device found", flush=True)


def load_camera_class():
    try:
        from RealSenseCamera import RealSenseCamera
    except Exception as exc:
        raise RuntimeError(
            "Failed to import RealSenseCamera.py. The viewer and camera module must "
            f"come from the same robot checkout. Original error: {exc}"
        ) from exc
    return RealSenseCamera


def construct_camera(camera_class, args: argparse.Namespace):
    """Construct old and new RealSenseCamera implementations safely."""
    kwargs: dict[str, Any] = {
        "width": args.width,
        "height": args.height,
        "fps": args.camera_fps,
    }
    try:
        parameters = inspect.signature(camera_class).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_extra = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    optional_values = {
        "enable_imu": bool(args.use_imu),
        "accel_fps": int(args.accel_fps),
        "gyro_fps": int(args.gyro_fps),
    }
    for name, value in optional_values.items():
        if accepts_extra or name in parameters:
            kwargs[name] = value

    if args.use_imu and "enable_imu" not in kwargs:
        raise RuntimeError(
            "This RealSenseCamera.py does not support optional IMU capture. Copy the "
            "matching RealSenseCamera.py from the robot or run without --use-imu."
        )
    return camera_class(**kwargs)


def sensor_qos_profile():
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def control_qos_profile():
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


class RealSenseViewerNode(Node):
    """ROS 2 node that publishes color/depth frames and optional IMU data."""

    def __init__(self, args: argparse.Namespace, camera_class):
        super().__init__("realsense_viewer_node")
        self.args = args
        self.running = True
        self.shutdown_requested = False
        self.fatal_error: str | None = None
        self._camera_stopped = False
        self._consecutive_errors = 0
        self._published_frames = 0
        self._status_started = time.monotonic()
        self._last_warning_time = 0.0

        if args.use_imu:
            check_realsense_imu_permission()
        self.camera = construct_camera(camera_class, args)

        sensor_qos = sensor_qos_profile()
        self.rgb_pub = self.create_publisher(
            Image, "/camera/color/image_raw", sensor_qos
        )
        self.depth_pub = self.create_publisher(
            Image, "/camera/depth/image_rect_raw", sensor_qos
        )
        self.camera_info_pub = self.create_publisher(
            CameraInfo, "/camera/camera_info", sensor_qos
        )
        self.depth_scale_pub = self.create_publisher(
            Float32, "/camera/depth_scale", sensor_qos
        )
        self.imu_pub = (
            self.create_publisher(Imu, "/camera/imu", sensor_qos)
            if args.use_imu
            else None
        )
        self.command_sub = self.create_subscription(
            String,
            "/camera/command",
            self.command_callback,
            control_qos_profile(),
        )

        self.timer = self.create_timer(1.0 / args.publish_fps, self.publish_loop)
        self.tf_broadcaster = None
        if args.publish_static_tf:
            if tf2_ros is None or TransformStamped is None:
                raise RuntimeError("--publish-static-tf requires tf2_ros")
            self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
            self._publish_static_tf()
            self.get_logger().warning(
                "Publishing an identity static TF. Do not use it as a hand-eye "
                "calibration for robot IK."
            )

        imu_description = "/camera/imu" if args.use_imu else "disabled"
        self.get_logger().info(
            "RealSense viewer node started. "
            f"domain={os.environ.get('ROS_DOMAIN_ID', '0')}, "
            f"rmw={os.environ.get('RMW_IMPLEMENTATION', 'default')}, "
            f"transport={os.environ.get('FASTDDS_BUILTIN_TRANSPORTS', 'default')}, "
            f"capture={args.width}x{args.height}@{args.camera_fps}, "
            f"publish={args.publish_fps:g}Hz, imu={imu_description}"
        )
        self.get_logger().info(
            "Publish: /camera/color/image_raw, /camera/depth/image_rect_raw, "
            "/camera/camera_info, /camera/depth_scale; "
            "Subscribe: /camera/command"
        )

    def command_callback(self, msg) -> None:
        command = (msg.data or "").strip().lower()
        if command in {"start", "resume", "run", "on"}:
            self.running = True
            self.get_logger().info("Camera streaming resumed")
        elif command in {"stop", "pause", "off"}:
            self.running = False
            self.get_logger().info("Camera streaming paused")
        elif command in {"exit", "quit", "shutdown"}:
            self.running = False
            self.shutdown_requested = True
            self.timer.cancel()
            self.get_logger().info("Camera node shutdown requested")
        elif command:
            self.get_logger().warning(f"Unknown camera command: {command!r}")

    def _publish_static_tf(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.args.static_parent_frame
        transform.child_frame_id = self.args.frame_id
        transform.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(transform)

    def _warn_throttled(self, message: str, *, period: float = 1.0) -> None:
        now = time.monotonic()
        if now - self._last_warning_time >= period:
            self.get_logger().warning(message)
            self._last_warning_time = now

    def _record_failure(self, exc: Exception) -> None:
        self._consecutive_errors += 1
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        self._warn_throttled(
            "Failed to capture/publish frame "
            f"({self._consecutive_errors}/{self.args.max_consecutive_errors}): {detail}"
        )
        if self._consecutive_errors >= self.args.max_consecutive_errors:
            self.fatal_error = (
                f"RealSense failed {self._consecutive_errors} consecutive times: {detail}"
            )
            self.shutdown_requested = True
            self.timer.cancel()
            self.get_logger().error(self.fatal_error)

    def publish_loop(self) -> None:
        if not self.running or self.shutdown_requested or not rclpy.ok():
            return

        try:
            frame = self.camera.get_frame()
            if frame is None:
                self._warn_throttled("RealSense returned no synchronized RGB-D frame")
                return

            color = np.asarray(frame["color"])
            depth = np.asarray(frame["depth"])
            if color.ndim != 3 or color.shape[2] != 3:
                raise ValueError(f"invalid BGR image shape: {color.shape}")
            if depth.ndim != 2:
                raise ValueError(f"invalid depth image shape: {depth.shape}")
            if color.shape[:2] != depth.shape[:2]:
                raise ValueError(
                    f"RGB/depth size mismatch: {color.shape[:2]} vs {depth.shape[:2]}"
                )

            now = self.get_clock().now().to_msg()
            self.rgb_pub.publish(self._to_image_msg(color, "bgr8", now))
            self.depth_pub.publish(self._to_image_msg(depth, "16UC1", now))
            self.camera_info_pub.publish(self._camera_info_message(color.shape, now))

            depth_scale_msg = Float32()
            depth_scale_msg.data = float(
                frame.get("depth_scale", self.camera.depth_scale)
            )
            self.depth_scale_pub.publish(depth_scale_msg)

            if self.imu_pub is not None and frame.get("imu") is not None:
                imu_message = self._imu_message(frame["imu"], now)
                if imu_message is not None:
                    self.imu_pub.publish(imu_message)

            self._consecutive_errors = 0
            self._published_frames += 1
            elapsed = time.monotonic() - self._status_started
            if self._published_frames == 1:
                self.get_logger().info(
                    f"First synchronized RGB-D frame published: "
                    f"{color.shape[1]}x{color.shape[0]}"
                )
            elif elapsed >= 10.0:
                rate = self._published_frames / elapsed
                self.get_logger().info(f"RGB-D publish rate: {rate:.2f} Hz")
                self._published_frames = 0
                self._status_started = time.monotonic()
        except Exception as exc:
            self._record_failure(exc)

    def _to_image_msg(self, image: np.ndarray, encoding: str, stamp):
        if encoding == "16UC1":
            image = np.asarray(image, dtype=np.uint16)
        else:
            image = np.asarray(image, dtype=np.uint8)
        image = np.ascontiguousarray(image)

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.args.frame_id
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = sys.byteorder == "big"
        msg.step = int(image.strides[0])
        msg.data = image.tobytes()
        return msg

    def _camera_info_message(self, color_shape, stamp):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.args.frame_id
        msg.height = int(color_shape[0])
        msg.width = int(color_shape[1])
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0] * 5
        msg.k = [
            float(self.camera.fx),
            0.0,
            float(self.camera.cx),
            0.0,
            float(self.camera.fy),
            float(self.camera.cy),
            0.0,
            0.0,
            1.0,
        ]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [
            float(self.camera.fx),
            0.0,
            float(self.camera.cx),
            0.0,
            0.0,
            float(self.camera.fy),
            float(self.camera.cy),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return msg

    def _imu_message(self, imu_data, stamp):
        gyro = np.asarray(imu_data["gyro"], dtype=np.float64).reshape(-1)
        accel = np.asarray(imu_data["accel"], dtype=np.float64).reshape(-1)
        if gyro.shape != (3,) or accel.shape != (3,):
            raise ValueError(
                f"invalid IMU vector shape: gyro={gyro.shape}, accel={accel.shape}"
            )
        if not np.all(np.isfinite(gyro)) or not np.all(np.isfinite(accel)):
            raise ValueError("IMU contains non-finite values")

        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self.args.imu_frame_id or self.args.frame_id
        msg.orientation_covariance[0] = -1.0
        msg.angular_velocity.x = float(gyro[0])
        msg.angular_velocity.y = float(gyro[1])
        msg.angular_velocity.z = float(gyro[2])
        msg.linear_acceleration.x = float(accel[0])
        msg.linear_acceleration.y = float(accel[1])
        msg.linear_acceleration.z = float(accel[2])
        return msg

    def close(self) -> None:
        if self._camera_stopped:
            return
        self._camera_stopped = True
        try:
            self.timer.cancel()
        except Exception:
            pass
        try:
            self.camera.stop()
        except Exception as exc:
            self.get_logger().warning(f"Failed to stop RealSense cleanly: {exc}")


def run_opencv_viewer(args: argparse.Namespace, camera_class) -> int:
    if not os.environ.get("DISPLAY"):
        print("ERROR: --opencv-only requires DISPLAY (for SSH, connect with ssh -Y).")
        return 3
    try:
        import cv2
    except Exception as exc:
        print(f"ERROR: OpenCV GUI is unavailable: {exc}")
        return 3

    camera = construct_camera(camera_class, args)
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
            color = np.asarray(frame["color"])
            depth_vis = camera.colorize_depth(frame["depth"])
            try:
                combined = cv2.hconcat([color, depth_vis])
            except Exception:
                combined = color
            cv2.imshow("RGB | Depth", combined)
            if cv2.waitKey(1) & 0xFF == 27:
                return 0
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def acquire_lock() -> bool:
    global _lock_file
    _lock_file = open(LOCKFILE_PATH, "w+")
    try:
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            "ERROR: another run_realsense_viewer.py instance is already running. "
            f"Lock file: {LOCKFILE_PATH}",
            flush=True,
        )
        _lock_file.close()
        _lock_file = None
        return False
    _lock_file.write(str(os.getpid()))
    _lock_file.flush()
    return True


def release_lock() -> None:
    global _lock_file
    if _lock_file is not None:
        try:
            fcntl.flock(_lock_file, fcntl.LOCK_UN)
            _lock_file.close()
        except Exception:
            pass
        _lock_file = None


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args, ros_args = parser.parse_known_args(argv)

    if not acquire_lock():
        return 2

    node = None
    ros_initialized = False
    exit_code = 0
    try:
        if not args.opencv_only and (rclpy is None or Image is None or String is None):
            print(
                "ERROR: ROS 2 Python packages are unavailable. Source "
                "/opt/ros/humble/setup.bash before starting this publisher.",
                flush=True,
            )
            if ROS_IMPORT_ERROR is not None:
                print(f"ROS import error: {ROS_IMPORT_ERROR}", flush=True)
            exit_code = 3
            return exit_code

        camera_class = load_camera_class()
        if args.opencv_only:
            exit_code = run_opencv_viewer(args, camera_class)
            return exit_code

        rclpy.init(args=ros_args)
        ros_initialized = True
        node = RealSenseViewerNode(args, camera_class)
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.25)
        if node.fatal_error is not None:
            exit_code = 1
    except KeyboardInterrupt:
        print("Interrupted by user", flush=True)
    except BaseException as exc:
        exit_code = 1
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        if node is not None:
            node.close()
            try:
                node.destroy_node()
            except Exception:
                pass
        if ros_initialized and rclpy.ok():
            rclpy.shutdown()
        release_lock()
        print(f"RealSense viewer stopped (exit={exit_code})", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
