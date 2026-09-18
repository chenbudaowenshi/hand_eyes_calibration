#!/usr/bin/env python3
"""Convert a raw-joint-angle handeye capture (images/pose_NN.png +
manifest.json, produced by a PDO-based capture tool that bypasses ROS2 TF)
into the rgb/*.png + poses.csv dataset layout that calib.cpp consumes.

Forward-kinematics parameters (joint origins/axes) are taken from the
delivered URDF, not hardcoded, so re-running against a different capture
session with a different URDF just needs --urdf pointed at the right file.

Usage:
  python3 tools/manifest_to_dataset.py \
      --capture-dir handeye_2026-09-15 \
      --urdf handeye_2026-09-15/shuangbi20260803_righthand.urdf \
      --root-link waist_yaw_Link \
      --out-dataset dataset_20260915
"""
import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_extrinsic_xyz_to_matrix(rx, ry, rz):
    """Matches this project's RPY::toRotation3D() for (XYZ, EXTRINSIC):
    R = Rz(rz) * Ry(ry) * Rx(rx)."""
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def matrix_to_rpy_extrinsic_xyz(r):
    """Inverse of the above; matches RPY::toRPY()'s (XYZ, EXTRINSIC) branch
    in src/common/Math.cpp exactly (same atan2 formulas), so calib.cpp's
    Pose3D round-trips back to this rotation matrix bit-for-bit modulo
    floating point."""
    eps = 1e-9
    ry = math.atan2(-r[2, 0], math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    if math.pi / 2 - eps < ry < math.pi / 2 + eps:
        ry = math.pi / 2
        rx = math.atan2(-r[1, 2], r[1, 1])
        rz = 0.0
    elif -math.pi / 2 - eps < ry < -math.pi / 2 + eps:
        ry = -math.pi / 2
        rx = math.atan2(-r[1, 2], r[1, 1])
        rz = 0.0
    else:
        c2 = math.cos(ry)
        rz = math.atan2(r[1, 0] / c2, r[0, 0] / c2)
        rx = math.atan2(r[2, 1] / c2, r[2, 2] / c2)
    return rx, ry, rz


def homogeneous(r, t):
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


class UrdfJoint:
    def __init__(self, elem):
        self.name = elem.get("name")
        self.type = elem.get("type")
        self.parent = elem.find("parent").get("link")
        self.child = elem.find("child").get("link")
        origin = elem.find("origin")
        xyz = origin.get("xyz") if origin is not None else "0 0 0"
        rpy = origin.get("rpy") if origin is not None else "0 0 0"
        self.origin_xyz = np.array([float(v) for v in xyz.split()])
        self.origin_rpy = np.array([float(v) for v in rpy.split()])
        axis = elem.find("axis")
        self.axis = (
            np.array([float(v) for v in axis.get("xyz").split()])
            if axis is not None
            else np.array([1.0, 0.0, 0.0])
        )

    def transform(self, angle=0.0):
        r_origin = rpy_extrinsic_xyz_to_matrix(*self.origin_rpy)
        t_origin = homogeneous(r_origin, self.origin_xyz)
        if self.type == "fixed":
            return t_origin
        axis = self.axis / np.linalg.norm(self.axis)
        # Rodrigues' rotation formula about an arbitrary axis (URDF axis is
        # not always a pure X/Y/Z unit vector in general, though it is here).
        k = axis
        kx, ky, kz = k
        kmat = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        r_axis = np.eye(3) + math.sin(angle) * kmat + (1 - math.cos(angle)) * (kmat @ kmat)
        return t_origin @ homogeneous(r_axis, np.zeros(3))


def load_joints(urdf_path):
    root = ET.parse(urdf_path).getroot()
    return {j.get("name"): UrdfJoint(j) for j in root.findall("joint")}


def find_chain(joints, root_link, target_link):
    """Returns the ordered list of joints from root_link down to
    target_link, following parent->child links only (this robot's chain
    from waist_yaw_Link to head_pitch_Link is a simple linear chain, no
    branching, so a straightforward walk is enough)."""
    by_parent = {}
    for j in joints.values():
        by_parent.setdefault(j.parent, []).append(j)

    chain = []
    current = root_link
    while current != target_link:
        candidates = by_parent.get(current, [])
        next_joint = None
        for j in candidates:
            # Prefer the branch that can actually reach target_link.
            if j.child == target_link:
                next_joint = j
                break
        if next_joint is None:
            for j in candidates:
                next_joint = j
                break
        if next_joint is None:
            raise RuntimeError(f"No path from {root_link} to {target_link} "
                               f"(stuck at {current})")
        chain.append(next_joint)
        current = next_joint.child
    return chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--root-link", default="waist_yaw_Link")
    ap.add_argument("--target-link", default="head_pitch_Link")
    ap.add_argument("--out-dataset", required=True)
    ap.add_argument("--file-prefix", default="",
                    help="Prepended to output image names to avoid "
                         "collisions when merging multiple capture "
                         "sessions (which all name frames pose_00.png...) "
                         "into one --out-dataset directory.")
    ap.add_argument("--append", action="store_true",
                    help="Append to an existing --out-dataset's poses.csv "
                         "instead of overwriting it (skips the header "
                         "line). Use with distinct --file-prefix values "
                         "per session.")
    args = ap.parse_args()

    capture_dir = Path(args.capture_dir)
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    joints = load_joints(args.urdf)
    chain = find_chain(joints, args.root_link, args.target_link)
    print(f"[INFO] FK chain {args.root_link} -> {args.target_link}: "
          + " -> ".join(j.name for j in chain))

    out_dir = Path(args.out_dataset)
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)

    csv_lines = []
    if not args.append:
        csv_lines.append(
            f"# image_name,capture_stamp_s,x_mm,y_mm,z_mm,rx_deg,ry_deg,rz_deg "
            f"-- T_{args.root_link}_{args.target_link}, RPY extrinsic X-Y-Z (deg)"
        )
    csv_lines.append(
        f"# {capture_dir} converted via URDF FK ({args.urdf})"
    )
    for record in manifest["records"]:
        q = record["q_ours"]
        t = np.eye(4)
        for j in chain:
            angle = q.get(j.name, 0.0) if j.type != "fixed" else 0.0
            t = t @ j.transform(angle)

        r = t[:3, :3]
        pos_mm = t[:3, 3] * 1000.0
        rx, ry, rz = matrix_to_rpy_extrinsic_xyz(r)
        rx_deg, ry_deg, rz_deg = math.degrees(rx), math.degrees(ry), math.degrees(rz)

        src_image = capture_dir / record["image"]
        image_name = args.file_prefix + src_image.name
        dst_image = out_dir / "rgb" / image_name
        shutil.copy2(src_image, dst_image)

        csv_lines.append(
            f"{image_name},{record['captured_monotonic']:.6f},"
            f"{pos_mm[0]:.6f},{pos_mm[1]:.6f},{pos_mm[2]:.6f},"
            f"{rx_deg:.6f},{ry_deg:.6f},{rz_deg:.6f}"
        )

    csv_path = out_dir / "poses.csv"
    mode = "a" if args.append else "w"
    with open(csv_path, mode) as f:
        f.write("\n".join(csv_lines) + "\n")

    cam = manifest["camera_info"]
    fs = out_dir / "realsense_intrinsics.xml"
    if args.append and fs.exists():
        print(f"[INFO] --append: leaving existing {fs} untouched "
              f"(camera_info assumed unchanged across sessions).")
    else:
        fs.write_text(
        "<?xml version=\"1.0\"?>\n<opencv_storage>\n"
        "<K type_id=\"opencv-matrix\">\n  <rows>3</rows>\n  <cols>3</cols>\n"
        "  <dt>d</dt>\n  <data>\n    "
        + " ".join(f"{v:.10g}" for v in cam["K"]) + "</data></K>\n"
        "<distortion type_id=\"opencv-matrix\">\n  <rows>1</rows>\n  <cols>5</cols>\n"
        "  <dt>d</dt>\n  <data>\n    "
        + " ".join(f"{v:.10g}" for v in cam["D"]) + "</data></distortion>\n"
        f"<width>{cam['width']}</width>\n<height>{cam['height']}</height>\n"
        "<note>\"Copied from manifest.json camera_info (live RealSense "
        "values recorded at PDO-capture time; no fps/depth_scale field in "
        "this manifest).\"</note>\n</opencv_storage>\n"
    )

    print(f"[INFO] Wrote {len(manifest['records'])} samples to {out_dir}")
    print(f"[INFO]   {out_dir}/rgb/*.png")
    print(f"[INFO]   {out_dir}/poses.csv")
    print(f"[INFO]   {out_dir}/realsense_intrinsics.xml")


if __name__ == "__main__":
    main()
