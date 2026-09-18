#!/usr/bin/env python3
"""Offline PARK-initialized head/camera calibration, with reserved validation.

For a two-joint chain, first/last joint zero offsets are gauge freedoms when
both camera and board poses are unknown. Zero-centered priors fix that gauge;
they do not turn it into a measurement of the physical encoder zero offsets.
All lengths in the solve are mm. Input images and robot files are read only.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import cv2
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from manifest_to_dataset import load_joints


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n")


def pack(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def unpack(p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
    T[:3, 3] = p[3:6]
    return T


def mean_pose(Ts):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_matrix(Ts[:, :3, :3]).mean().as_matrix()
    T[:3, 3] = Ts[:, :3, 3].mean(0)
    return T


def statistics(x):
    x = np.asarray(x)
    if not x.size:
        return {"count": 0, "rms": None, "mean": None, "max": None}
    return {"count": int(x.size), "rms": float(np.sqrt(np.mean(x*x))),
            "mean": float(np.mean(x)), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "max": float(np.max(x))}


class HeadKinematics:
    def __init__(self, urdf, records):
        joints = load_joints(urdf)
        self.names = ["head_yaw_joint", "head_pitch_joint"]
        self.joints = [joints[name] for name in self.names]
        if (self.joints[0].parent != "waist_yaw_Link"
                or self.joints[0].child != self.joints[1].parent
                or self.joints[1].child != "head_pitch_Link"):
            raise ValueError("Expected waist_yaw_Link -> head_yaw -> head_pitch chain")
        if any(j.type not in ("revolute", "continuous") for j in self.joints):
            raise ValueError("Only the two revolute head joints are supported")
        self.origins = [j.transform(0) for j in self.joints]
        for O in self.origins:
            O[:3, 3] *= 1000
        self.axes = [j.axis / np.linalg.norm(j.axis) for j in self.joints]
        self.q = np.array([[r["q_ours"][name] for name in self.names] for r in records])
        if not np.isfinite(self.q).all():
            raise ValueError("Non-finite measured joints")

    def rotation(self, axis, angles):
        angles = np.atleast_1d(angles)
        A = np.tile(np.eye(4), (len(angles), 1, 1))
        A[:, :3, :3] = Rotation.from_rotvec(angles[:, None]*axis).as_matrix()
        return A

    def forward(self, offsets=(0., 0.)):
        return (self.origins[0] @ self.rotation(self.axes[0], self.q[:, 0]+offsets[0])
                @ self.origins[1] @ self.rotation(self.axes[1], self.q[:, 1]+offsets[1]))

    def canonical(self, X, Y, offsets):
        """Absorb endpoint offsets into X/Y; express result at nominal q."""
        pitch = self.rotation(self.axes[1], [offsets[1]])[0]
        yaw = self.rotation(self.axes[0], [offsets[0]])[0]
        B = self.origins[0] @ yaw @ np.linalg.inv(self.origins[0])
        return pitch @ X, np.linalg.inv(B) @ Y


def detect(args, records, K, D, obj):
    root, out = args.capture, args.output
    images = [root/r["image"] for r in records]
    image_hashes = [digest(p) for p in images]
    if len(set(image_hashes)) != len(image_hashes):
        raise ValueError("Duplicate image contents: cannot claim independent observations")
    key = {"manifest_sha256": digest(root/"manifest.json"),
           "image_sha256": image_hashes, "opencv": cv2.__version__,
           "detector": "SB_EXHAUSTIVE_ACCURACY_top_left_endpoint",
           "cols": args.cols, "rows": args.rows, "square_mm": args.square_mm,
           "corner_refinement_half_window": args.corner_refinement,
           "training_end": args.training_end}
    cache, meta = out/"corners.npz", out/"corner_cache_metadata.json"
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == key:
        print("Reusing checked corner cache", flush=True)
        data = np.load(cache)
        return data["corners"], json.loads((out/"detection.json").read_text())
    corners = np.full((len(records), len(obj), 2), np.nan)
    report = []
    for i, path in enumerate(images):
        raw = cv2.imread(str(path))
        if raw is None:
            raise ValueError(f"Cannot read {path}")
        if raw.shape[:2] != (args.height, args.width):
            raise ValueError(f"Image size differs from camera_info: {path}, {raw.shape}")
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        found, pc = cv2.findChessboardCornersSB(
            gray, (args.cols, args.rows),
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        item = {"index": i, "pose_index": records[i]["pose_index"],
                "image": str(path.relative_to(root)), "found": bool(found),
                "split": "train" if i < args.training_end else "validation"}
        if found:
            pc = pc.reshape(-1, 2).astype(float)
            # Dataset is a fixed upright board with small camera roll. This
            # defines one consistent physical endpoint across the session;
            # it is not a general replacement for coded target identities.
            if pc[0].sum() > pc[-1].sum():
                pc = pc[::-1].copy()
            if args.corner_refinement:
                refined = pc.astype(np.float32).reshape(-1, 1, 2)
                cv2.cornerSubPix(gray, refined,
                                (args.corner_refinement, args.corner_refinement),
                                (-1, -1), (3, 60, 1e-5))
                pc = refined.reshape(-1, 2).astype(float)
            corners[i] = pc
            grid = pc.reshape(args.rows, args.cols, 2)
            steps = np.r_[np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
                          np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()]
            ok, rv, tv = cv2.solvePnP(obj, pc, K, D)
            if not ok:
                raise RuntimeError(f"PnP failed for {path}")
            projected = cv2.projectPoints(obj, rv, tv, K, D)[0].reshape(-1, 2)
            error = np.linalg.norm(projected-pc, axis=1)
            item.update({"spacing_min_px": float(steps.min()),
                         "spacing_median_px": float(np.median(steps)),
                         "board_span_xy_px": np.ptp(pc, axis=0).tolist(),
                         "pnp_reprojection_px": statistics(error),
                         "pnp_z_mm": float(tv[2, 0])})
            if i in (0, args.training_end, len(records)-1):
                vis = raw.copy()
                cv2.drawChessboardCorners(vis, (args.cols, args.rows),
                                         pc.astype(np.float32).reshape(-1, 1, 2), True)
                cv2.imwrite(str(out/f"detected_{i:03d}.png"), vis)
        report.append(item)
        if i % 10 == 0:
            print("Detected", i+1, "/", len(images), "last found", found, flush=True)
    np.savez_compressed(cache, corners=corners)
    write_json(meta, key)
    write_json(out/"detection.json", report)
    return corners, report


def project(p, kin, indices, obj, K, D):
    X, Y = unpack(p[:6]), unpack(p[6:12])
    G = kin.forward(p[12:14] if len(p) == 14 else (0., 0.))
    C = np.linalg.inv(X) @ np.linalg.inv(G[indices]) @ Y
    points = []
    for c in C:
        # Rodrigues avoids Euler singularities; raw corners use raw K/D once.
        pp = cv2.projectPoints(obj, cv2.Rodrigues(c[:3, :3])[0], c[:3, 3], K, D)[0]
        points.append(pp.reshape(-1, 2))
    return np.asarray(points)


def evaluate(label, p, kin, ids, C, corners, obj, K, D, training_end):
    offsets = np.asarray(p[12:14]) if len(p) == 14 else np.zeros(2)
    X, Y = unpack(p[:6]), unpack(p[6:12])
    G = kin.forward(offsets)
    Ys = G[ids] @ X @ C
    train = ids < training_end
    # Reference center and reference orientation are train-only.
    center = Ys[train, :3, 3].mean(0)
    errors = np.linalg.norm(Ys[:, :3, 3]-center, axis=1)
    errors_y = np.linalg.norm(Ys[:, :3, 3]-Y[:3, 3], axis=1)
    clouds = np.einsum("nij,pj->npi", Ys[:, :3, :3], obj)+Ys[:, None, :3, 3]
    target = obj @ Y[:3, :3].T+Y[:3, 3]
    corners_mm = np.linalg.norm(clouds-target, axis=2)
    pixels = np.linalg.norm(project(p, kin, ids, obj, K, D)-corners[ids], axis=2)
    rotation_errors = Rotation.from_matrix(Ys[:, :3, :3] @ Y[:3, :3].T).magnitude()*180/np.pi
    Xc, Yc = kin.canonical(X, Y, offsets)
    out = {"method": label, "offsets_yaw_pitch_deg": np.degrees(offsets).tolist(),
           "offsets_are_physical_estimates": False,
           "X_head_camera_optical_mm": X.tolist(), "Y_base_board_mm": Y.tolist(),
           "X_nominal_head_camera_optical_mm": Xc.tolist(),
           "Y_nominal_base_board_mm": Yc.tolist(),
           "train_reference_origin_mm": center.tolist()}
    rows = []
    for name, mask in (("train", train), ("validation", ~train)):
        xyz = Ys[mask, :3, 3]-center
        out[name] = {"pose_count": int(mask.sum()),
                     "origin_mm": statistics(errors[mask]),
                     "origin_to_optimized_Y_mm": statistics(errors_y[mask]),
                     "all_corners_to_Y_mm": statistics(corners_mm[mask]),
                     "reprojection_px": statistics(pixels[mask]),
                     "board_rotation_deg": statistics(rotation_errors[mask]),
                     "axis_rms_mm": np.sqrt(np.mean(xyz*xyz, axis=0)).tolist()}
    for j, i in enumerate(ids):
        rows.append({"index": int(i), "split": "train" if train[j] else "validation",
                     "origin_error_mm": float(errors[j]), "origin_to_Y_mm": float(errors_y[j]),
                     "corner_rms_mm": float(np.sqrt(np.mean(corners_mm[j]**2))),
                     "reprojection_rms_px": float(np.sqrt(np.mean(pixels[j]**2))),
                     "board_rotation_error_deg": float(rotation_errors[j])})
    out["per_frame"] = rows
    print(label, "offsets(deg)", np.degrees(offsets), flush=True)
    for split in ("train", "validation"):
        m = out[split]
        print(split, "n", m["pose_count"], "origin RMS/Max mm",
              m["origin_mm"]["rms"], m["origin_mm"]["max"],
              "all-corner RMS", m["all_corners_to_Y_mm"]["rms"],
              "px RMS", m["reprojection_px"]["rms"], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, default=Path("tools/reference_urdf/shuangbi20260803.urdf"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--validate-count", type=int, default=20)
    ap.add_argument("--cols", type=int, default=11)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--square-mm", type=float, default=15.)
    ap.add_argument("--corner-refinement", type=int, default=0,
                    help="Optional cornerSubPix half-window; 4 repeats the prior tested workflow")
    ap.add_argument("--objective", choices=("reprojection", "fullboard"), default="reprojection",
                    help="Primary objective fixed before examining validation results")
    args = ap.parse_args()
    if args.cols < 2 or args.rows < 2 or not np.isfinite(args.square_mm) or args.square_mm <= 0:
        raise ValueError("Invalid board dimensions or square size")
    if args.corner_refinement < 0:
        raise ValueError("corner-refinement must be nonnegative")
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.capture/"manifest.json").read_text())
    records = sorted(manifest["records"], key=lambda r: r["pose_index"])
    if len(set(r["pose_index"] for r in records)) != len(records):
        raise ValueError("Duplicate pose indices")
    if not 1 <= args.validate_count <= len(records)-10:
        raise ValueError("Need >=10 training poses and >=1 validation pose")
    args.training_end = len(records)-args.validate_count
    cam = manifest["camera_info"]
    args.width, args.height = cam["width"], cam["height"]
    K = np.array(cam["K"], dtype=float).reshape(3, 3)
    D = np.array(cam["D"], dtype=float)
    if cam["distortion_model"] != "plumb_bob" or not np.isfinite(K).all() or not np.isfinite(D).all():
        raise ValueError("Unsupported or nonfinite camera model")
    obj = np.zeros((args.cols*args.rows, 3), dtype=float)
    obj[:, :2] = np.mgrid[:args.cols, :args.rows].T.reshape(-1, 2)*args.square_mm
    kin = HeadKinematics(args.urdf, records)
    direction = manifest["conventions"]["hardware_direction"]
    for r in records:
        for name in kin.names:
            if abs(r["q_ours"][name]-r["q_raw"][name]*direction[name]) > 1e-8:
                raise ValueError("q_raw/q_ours mismatch")
    write_json(args.output/"split.json", {
        "rule": "Sorted original pose_index; last N reserved before detection/optimization",
        "training_pose_indices": [r["pose_index"] for r in records[:args.training_end]],
        "validation_pose_indices": [r["pose_index"] for r in records[args.training_end:]],
        "manifest_sha256": digest(args.capture/"manifest.json"),
        "urdf_sha256": digest(args.urdf), "square_mm": args.square_mm,
        "analysis_plan": {"primary_objective": args.objective,
                          "corner_refinement_half_window": args.corner_refinement,
                          "intrinsics": "fixed manifest K/D",
                          "initialization": "PARK then Huber(1 px) reprojection",
                          "fullboard_loss": "linear" if args.objective == "fullboard" else None,
                          "zero_offset_gauge_prior_deg": 1.0,
                          "remove_frames_based_on_fit_residual": False}})
    corners, detection = detect(args, records, K, D, obj)
    ids = np.flatnonzero(np.isfinite(corners).all((1, 2)))
    tr = ids[ids < args.training_end]
    if len(tr) < 10 or not np.any(ids >= args.training_end):
        raise ValueError("Insufficient detected train/validation images")
    C = []
    for i in ids:
        ok, r, t = cv2.solvePnP(obj, corners[i], K, D)
        c = np.eye(4); c[:3, :3] = cv2.Rodrigues(r)[0]; c[:3, 3] = t.ravel()
        if not ok or np.min(obj @ c[2, :3]+c[2, 3]) <= 0:
            raise ValueError("PnP failed/points behind camera")
        C.append(c)
    C = np.asarray(C)
    G = kin.forward()
    mask = ids < args.training_end
    r, t = cv2.calibrateHandEye(G[tr, :3, :3], G[tr, :3, 3], C[mask, :3, :3],
                              C[mask, :3, 3], method=cv2.CALIB_HAND_EYE_PARK)
    X = np.eye(4); X[:3, :3] = r; X[:3, 3] = t.ravel()
    if not np.isfinite(X).all() or abs(np.linalg.det(r)-1) > 1e-6:
        raise ValueError("PARK failed to produce a proper finite rotation")
    Y = mean_pose(G[tr] @ X @ C[mask])
    p0 = np.r_[pack(X), pack(Y)]
    baseline = evaluate("PARK", p0, kin, ids, C, corners, obj, K, D, args.training_end)
    # Fixed policy: Huber at 1 px, no per-frame corrections, no residual-based
    # image deletion, no intrinsic fitting, no validation-based model selection.
    def reprojection_residual(p):
        return (project(p, kin, tr, obj, K, D)-corners[tr]).ravel()
    fit12 = least_squares(reprojection_residual, p0, loss="huber", f_scale=1., x_scale="jac",
                          max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-8)
    model12 = evaluate("PARK + joint reprojection (nominal joints)", fit12.x,
                       kin, ids, C, corners, obj, K, D, args.training_end)
    data_residual = reprojection_residual
    fitting_loss = "huber"
    offset_start = fit12.x
    model3d = None
    if args.objective == "fullboard":
        def fullboard_residual(p):
            Xp, Yp = unpack(p[:6]), unpack(p[6:12])
            Gp = kin.forward(p[12:14] if len(p) == 14 else (0., 0.))
            Ys = Gp[tr] @ Xp @ C[mask]
            points = np.einsum("nij,pj->npi", Ys[:, :3, :3], obj) + Ys[:, None, :3, 3]
            target = obj @ Yp[:3, :3].T + Yp[:3, 3]
            return (points-target).ravel()
        data_residual = fullboard_residual
        fitting_loss = "linear"
        fit3d = least_squares(data_residual, fit12.x, loss=fitting_loss, x_scale="jac",
                              max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-8)
        offset_start = fit3d.x
        model3d = evaluate("PARK + full-board 3D optimization (nominal joints)",
                           fit3d.x, kin, ids, C, corners, obj, K, D, args.training_end)
    # Endpoint gauge fixing: weak physical-looking numbers are NOT estimates.
    # The 1-degree priors only choose one equivalent coordinate convention.
    def regularized(p):
        return np.r_[data_residual(p), p[12:14]/np.deg2rad(1.)]
    fit14 = least_squares(regularized, np.r_[offset_start, 0., 0.], loss=fitting_loss,
                          f_scale=1., x_scale="jac", max_nfev=500,
                          ftol=1e-10, xtol=1e-10, gtol=1e-8)
    final = evaluate("PARK + joint camera/board/zero-offset optimization (gauge-fixed, " + args.objective + ")",
                     fit14.x, kin, ids, C, corners, obj, K, D, args.training_end)
    # Numerical observation rank uses fixed image residual coordinates. With
    # a full-board 3D objective, changing the yaw gauge rotates the whole
    # nonzero residual vector without changing its norm, which can misleadingly
    # add a Jacobian direction. Image projections stay exactly invariant.
    steps = np.r_[np.full(3, 1e-6), np.full(3, 1e-3),
                  np.full(3, 1e-6), np.full(3, 1e-3), np.full(2, 1e-6)]
    scales = np.r_[np.ones(3), np.full(3, 100.), np.ones(3), np.full(3, 100.), np.ones(2)]
    cols = []
    for j, h in enumerate(steps):
        d = np.zeros(14); d[j] = h
        cols.append((reprojection_residual(fit14.x+d)-reprojection_residual(fit14.x-d))/(2*h)*scales[j])
    singular = np.linalg.svd(np.array(cols).T, compute_uv=False)
    Xc, Yc = kin.canonical(unpack(fit14.x[:6]), unpack(fit14.x[6:12]), fit14.x[12:14])
    gauge_projection = np.max(np.abs(project(np.r_[pack(Xc), pack(Yc)], kin, tr, obj, K, D)
                                     -project(fit14.x, kin, tr, obj, K, D)))
    def optimizer_info(res):
        return {"success": bool(res.success), "message": str(res.message),
                "nfev": res.nfev, "cost": float(res.cost), "optimality": float(res.optimality)}
    final["optimizer"] = optimizer_info(fit14)
    model12["optimizer"] = optimizer_info(fit12)
    result_list = [baseline, model12]
    if model3d is not None:
        model3d["optimizer"] = optimizer_info(fit3d)
        result_list.append(model3d)
    result_list.append(final)
    missing = [r for r in detection if not r["found"]]
    validation_complete = all(r["found"] for r in detection[args.training_end:])
    passed = bool(validation_complete and fit14.success and fit12.success
                  and (model3d is None or fit3d.success)
                  and final["validation"]["origin_mm"]["rms"] <= 5.)
    payload = {
        "capture": str(args.capture.resolve()), "urdf": str(args.urdf.resolve()),
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "solver_source_sha256": digest(__file__),
        "camera_info": cam, "intrinsics_provenance": "Manifest values, not independently SDK-verified",
        "primary_objective": args.objective,
        "corner_refinement_half_window": args.corner_refinement,
        "board": {"cols": args.cols, "rows": args.rows, "square_mm": args.square_mm},
        "joint_names": kin.names, "joint_ranges_deg": np.degrees(np.stack([kin.q.min(0), kin.q.max(0)])).tolist(),
        "training_count_requested": args.training_end, "validation_count_requested": args.validate_count,
        "detection_failures": missing, "validation_detection_complete": validation_complete,
        "goal": {"metric": "independent 3D board-origin consistency RMS to training-only mean", "threshold_mm": 5., "passed": passed},
        "joint_offset_identifiability": {
            "identifiable": False, "data_jacobian_scaled_singular_values": singular.tolist(),
            "jacobian_observation": "fixed image pixel coordinates, no priors",
            "rank_relative_tolerance_1e_7": int(np.sum(singular > singular[0]*1e-7)),
            "parameter_count": 14, "prior_sigma_deg": 1.,
            "canonical_projection_difference_max_px": float(gauge_projection),
            "explanation": "First joint zero rotates unknown board/base relation; last joint zero rotates camera extrinsics. Offsets fixed by priors, not measured."},
        "results": result_list,
        "limitations": ["One fixed board session assumed; no external absolute reference",
                        "No source exposure/PDO timestamp pair; synchronization cannot be audited",
                        f"{args.square_mm:g} mm is nominal user-provided square size",
                        "Same-session pose holdout, not a separately captured validation session",
                        "No physical joint-zero estimate is identifiable from this two-axis setup"]}
    write_json(args.output/"results.json", payload)
    for result, name in ((baseline, "park"), (final, "joint")):
        with (args.output/f"{name}_per_frame.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=result["per_frame"][0].keys())
            writer.writeheader(); writer.writerows(result["per_frame"])
    # Always save a candidate, with an explicit acceptance state. Never deploy.
    fs = cv2.FileStorage(str(args.output/"candidate_extrinsic.xml"), cv2.FILE_STORAGE_WRITE)
    fs.write("R", Xc[:3, :3]); fs.write("t", Xc[:3, 3]); fs.write("T_head_camera_optical", Xc)
    fs.write("translation_unit", "mm"); fs.write("joint_offsets_convention", "nominal q_ours; effective extrinsic")
    fs.write("validation_rms_mm", final["validation"]["origin_mm"]["rms"])
    fs.write("validation_passed", int(passed)); fs.release()
    np.savez_compressed(args.output/"observations.npz", G=G, q=kin.q, ids=ids, C=C,
                        corners=corners, obj=obj, K=K, D=D, fitted_parameters=fit14.x)
    print("RESULT:", "PASS" if passed else "NOT PASSED", "RMS <= 5 mm; report", args.output/"results.json", flush=True)


if __name__ == "__main__":
    main()
