"""Numerical invariants for the head calibration; no hardware access."""
import contextlib
import io
from pathlib import Path
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_head_joint import HeadKinematics, evaluate, pack, unpack, project


class HeadCalibrationInvariants(unittest.TestCase):
    def setUp(self):
        self.urdf = Path(__file__).resolve().parents[1]/"tools/reference_urdf/shuangbi20260803.urdf"
        rng = np.random.default_rng(4917)
        q = rng.uniform([-.35, -.15], [.6, .3], (30, 2))
        self.records = [{"q_ours": {"head_yaw_joint": a, "head_pitch_joint": b}} for a, b in q]
        self.kin = HeadKinematics(self.urdf, self.records)
        self.X = np.eye(4)
        self.X[:3, :3] = np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
        self.X[:3, 3] = [45., 70., -25.]
        self.Y = self.kin.forward()[0] @ self.X @ unpack([.12,-.05,.02,0.,0.,900.])
        self.obj = np.zeros((88,3))
        self.obj[:,:2] = np.mgrid[:11,:8].T.reshape(-1,2)*15.
        self.K = np.array([[911.,0.,636.],[0.,908.,380.],[0.,0.,1.]])
        self.D = np.zeros(5)

    def test_fk_matches_independent_per_joint_urdf_composition(self):
        batch = self.kin.forward([.02,-.03])
        for i, q in enumerate(self.kin.q):
            T = self.kin.joints[0].transform(q[0]+.02) @ self.kin.joints[1].transform(q[1]-.03)
            T[:3,3] *= 1000.
            np.testing.assert_allclose(batch[i],T,atol=1e-10)

    def test_endpoint_offsets_exactly_absorbed_by_camera_and_board(self):
        offsets = np.radians([4., -3.])
        X0, Y0 = self.kin.canonical(self.X,self.Y,offsets)
        C1 = np.linalg.inv(self.X) @ np.linalg.inv(self.kin.forward(offsets)) @ self.Y
        C0 = np.linalg.inv(X0) @ np.linalg.inv(self.kin.forward()) @ Y0
        np.testing.assert_allclose(C1,C0,atol=1e-10)
        p14=np.r_[pack(self.X),pack(self.Y),offsets]
        p12=np.r_[pack(X0),pack(Y0)]
        np.testing.assert_allclose(project(p14,self.kin,np.arange(30),self.obj,self.K,self.D),
                                   project(p12,self.kin,np.arange(30),self.obj,self.K,self.D),atol=1e-8)

    def test_park_recovers_planted_transform_from_exact_observations(self):
        G=self.kin.forward()
        C=np.linalg.inv(self.X) @ np.linalg.inv(G) @ self.Y
        r,t=cv2.calibrateHandEye(G[:,:3,:3],G[:,:3,3],C[:,:3,:3],C[:,:3,3],
                                method=cv2.CALIB_HAND_EYE_PARK)
        np.testing.assert_allclose(r,self.X[:3,:3],atol=1e-9)
        np.testing.assert_allclose(t.ravel(),self.X[:3,3],atol=1e-6)

    def test_noisy_fullboard_cost_is_invariant_to_offset_gauge(self):
        offsets = np.radians([4., -3.])
        X0, Y0 = self.kin.canonical(self.X, self.Y, offsets)
        rng = np.random.default_rng(718)
        C = np.linalg.inv(self.X) @ np.linalg.inv(self.kin.forward(offsets)) @ self.Y
        C[:, :3, 3] += rng.normal(0, 3, (len(C), 3))
        def errors(G, X, Y):
            Ys = G @ X @ C
            points = np.einsum('nij,pj->npi', Ys[:, :3, :3], self.obj) + Ys[:, None, :3, 3]
            return points - (self.obj @ Y[:3, :3].T + Y[:3, 3])
        a = errors(self.kin.forward(offsets), self.X, self.Y)
        b = errors(self.kin.forward(), X0, Y0)
        np.testing.assert_allclose(np.sum(a*a), np.sum(b*b), rtol=1e-12)

    def test_validation_cannot_move_training_reference(self):
        G=self.kin.forward();p=np.r_[pack(self.X),pack(self.Y)]
        C=np.linalg.inv(self.X) @ np.linalg.inv(G) @ self.Y
        pixels=project(p,self.kin,np.arange(30),self.obj,self.K,self.D)
        with contextlib.redirect_stdout(io.StringIO()):
            exact=evaluate("exact",p,self.kin,np.arange(30),C,pixels,self.obj,self.K,self.D,20)
            C[20:,:3,3] += [100.,0.,0.]
            shifted=evaluate("bad validation",p,self.kin,np.arange(30),C,pixels,self.obj,self.K,self.D,20)
        np.testing.assert_allclose(exact["train_reference_origin_mm"],shifted["train_reference_origin_mm"],atol=1e-12)
        self.assertLess(shifted["train"]["origin_mm"]["rms"],1e-8)
        self.assertAlmostEqual(shifted["validation"]["origin_mm"]["rms"],100.,places=8)


if __name__ == "__main__":
    unittest.main()
