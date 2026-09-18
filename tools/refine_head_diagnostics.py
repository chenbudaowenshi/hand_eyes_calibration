#!/usr/bin/env python3
"""Secondary diagnostics; frozen 80/20 split, all attempts reported.

These diagnostics follow the first holdout evaluation and therefore any new
passing pipeline would require new independent confirmation images.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from calibrate_head_joint import (HeadKinematics, pack, unpack, mean_pose,
                                 project, evaluate, write_json)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cv2.setNumThreads(2)
    obs = np.load(args.output/"observations.npz")
    results = json.loads((args.output/"results.json").read_text())
    records = sorted(json.loads((args.capture/"manifest.json").read_text())["records"],
                     key=lambda r: r["pose_index"])
    kin = HeadKinematics(results["urdf"], records)
    obj, K, D, ids = [obs[k] for k in ("obj", "K", "D", "ids")]
    end = results["training_count_requested"]
    mask = ids < end; tr = ids[mask]; G = kin.forward()
    outputs = []
    pc2 = obs["corners"].copy()
    # 9x9 support stays within ~13-15 px spacing; raw image only.
    for i in ids:
        gray = cv2.cvtColor(cv2.imread(str(args.capture/records[i]["image"])), cv2.COLOR_BGR2GRAY)
        pc = pc2[i].astype(np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(gray, pc, (4, 4), (-1, -1), (3, 60, 1e-5))
        pc2[i] = pc.reshape(-1, 2)
    np.savez_compressed(args.output/"refined_corners.npz", corners=pc2)
    for label, corners in (("SB", obs["corners"]), ("SB_subpix4", pc2)):
        C = []
        for i in ids:
            ok, rv, tv = cv2.solvePnP(obj, corners[i], K, D)
            assert ok
            c = np.eye(4); c[:3, :3] = cv2.Rodrigues(rv)[0]; c[:3, 3] = tv.ravel(); C.append(c)
        C = np.asarray(C)
        r, t = cv2.calibrateHandEye(G[tr, :3, :3], G[tr, :3, 3], C[mask, :3, :3],
                                  C[mask, :3, 3], method=cv2.CALIB_HAND_EYE_PARK)
        X0 = np.eye(4); X0[:3, :3] = r; X0[:3, 3] = t.ravel()
        p0 = np.r_[pack(X0), pack(mean_pose(G[tr] @ X0 @ C[mask]))]
        f = lambda p: (project(p, kin, tr, obj, K, D)-corners[tr]).ravel()
        fit = least_squares(f, p0, loss="huber", f_scale=1., x_scale="jac", max_nfev=500,
                            ftol=1e-10, xtol=1e-10, gtol=1e-8)
        outputs.append(evaluate(label+" reprojection", fit.x, kin, ids, C, corners, obj,K,D,end))
        def point_residual(p):
            X,Y=unpack(p[:6]),unpack(p[6:12])
            Ys=G[tr]@X@C[mask]
            pbs=np.einsum("nij,pj->npi",Ys[:,:3,:3],obj)+Ys[:,None,:3,3]
            target=obj@Y[:3,:3].T+Y[:3,3]
            return (pbs-target).ravel()
        fit3 = least_squares(point_residual, fit.x, loss="linear", x_scale="jac",
                             max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-8)
        outputs.append(evaluate(label+" full-board 3D least squares",fit3.x,
                                kin,ids,C,corners,obj,K,D,end))
        def origin_residual(p):
            X=unpack(p[:6]); target=p[6:9]
            Ys=G[tr]@X@C[mask]
            return (Ys[:,:3,3]-target).ravel()
        po=np.r_[fit.x[:6],unpack(fit.x[6:12])[:3,3]]
        fo=least_squares(origin_residual,po,loss="linear",x_scale="jac",max_nfev=500,
                         ftol=1e-10,xtol=1e-10,gtol=1e-8)
        Y=mean_pose(G[tr]@unpack(fo.x[:6])@C[mask]);Y[:3,3]=fo.x[6:9]
        outputs.append(evaluate(label+" board-origin-only 3D least squares diagnostic",
                                np.r_[fo.x[:6],pack(Y)],kin,ids,C,corners,obj,K,D,end))
        # Length in camera frame cannot be changed by the hand-eye rotation.
        # Hold out only the TRAIN observations to estimate the origin-floor.
        for item, solution in zip(outputs[-3:], (fit, fit3, fo)):
            item["evaluation_is_post_initial_holdout_diagnostic"] = True
            item["optimizer"] = {"success": bool(solution.success),
                                 "message": str(solution.message),
                                 "cost": float(solution.cost), "nfev": solution.nfev}
    write_json(args.output/"additional_diagnostics.json",{
        "note":"All trials listed; original validation indices unchanged; new confirmation needed if choosing a new model after these diagnostics.",
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
        "results":outputs})


if __name__ == "__main__":
    main()
