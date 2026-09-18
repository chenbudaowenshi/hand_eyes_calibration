#!/usr/bin/env python3
"""Export the preselected head calibration, preserving holdout scope and gauge."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    report = json.loads((out/"results.json").read_text())
    split = json.loads((out/"split.json").read_text())
    detection = json.loads((out/"detection.json").read_text())
    capture = Path(report["capture"])
    manifest = json.loads((capture/"manifest.json").read_text())
    final = report["results"][-1]
    X = np.array(final["X_nominal_head_camera_optical_mm"])
    Y = np.array(final["Y_nominal_base_board_mm"])
    angles_rad = Rotation.from_matrix(X[:3,:3]).as_euler("xyz")
    v = final["validation"]
    passed = report["goal"]["passed"]
    status = "PASSED_HELDOUT_ORIGIN_RMS_5MM" if passed else "NOT_PASSED_HELDOUT_ORIGIN_RMS_5MM"
    camera, board = report["camera_info"], report["board"]
    found = [d for d in detection if d["found"]]
    depths = np.array([d["pnp_z_mm"] for d in found])
    spacing = np.array([d["spacing_median_px"] for d in found])
    per_view_px = np.array([d["pnp_reprojection_px"]["rms"] for d in found])
    training_end = report["training_count_requested"]
    validation = [f for f in final["per_frame"] if f["split"] == "validation"]
    exceeded = [f["index"] for f in validation if f["origin_error_mm"] > 5.]
    observations = np.load(out/"observations.npz")
    nominal_board_poses = observations["G"][observations["ids"]] @ X @ observations["C"]
    center_nominal = nominal_board_poses[observations["ids"] < training_end, :3, 3].mean(0)
    accepted = {
        "status": status, "deployed": False, "method": final["method"],
        "coordinate_equation": "p_head_pitch_Link = R * p_RGB_optical + t",
        "parent_frame": "head_pitch_Link", "child_frame": "RGB_optical",
        "translation_unit": "mm", "angles": "fixed-axis XYZ",
        "joint_input": "measured q_ours with original hardware_direction conversion",
        "physical_joint_offsets_identifiable": False,
        "joint_offsets_to_apply_rad": [0., 0.],
        "offset_note": "Canonical effective extrinsic; do not add fitted gauge offsets to q_ours.",
        "T_head_pitch_RGB_optical_mm": X.tolist(),
        "translation_mm": X[:3,3].tolist(),
        "rpy_optical_deg": np.degrees(angles_rad).tolist(),
        "T_base_board_mm": Y.tolist(),
        "training_reference_origin_nominal_base_mm": center_nominal.tolist(),
        "validation": v, "training": final["train"],
        "input_provenance": {"manifest_sha256": split["manifest_sha256"],
                             "urdf_sha256": split["urdf_sha256"],
                             "solver_sha256": report["solver_source_sha256"]},
        "scope": {"same_session_holdout": True,
                  "pose_indices": split["validation_pose_indices"],
                  "observed_board_z_range_mm": [float(depths.min()),float(depths.max())],
                  "yaw_pitch_ranges_deg": report["joint_ranges_deg"],
                  "absolute_accuracy_verified": False,
                  "depth_pointcloud_verified": False,
                  "every_origin_error_le_5mm": len(exceeded)==0},
        "board": board, "camera_info": camera}
    (out/"calibration_result.json").write_text(json.dumps(accepted,ensure_ascii=False,indent=2)+"\n")
    fs = cv2.FileStorage(str(out/"calib_extrinsic.xml"),cv2.FILE_STORAGE_WRITE)
    fs.write("R",X[:3,:3]);fs.write("t",X[:3,3].reshape(3,1));fs.write("T_head_pitch_RGB_optical",X)
    fs.write("translation_unit","mm");fs.write("parent_frame","head_pitch_Link")
    fs.write("child_frame","RGB_optical");fs.write("status",status)
    fs.write("joint_offsets_to_apply_rad",np.zeros((2,1)))
    fs.write("validation_rms_mm",v["origin_mm"]["rms"])
    fs.write("validation_max_mm",v["origin_mm"]["max"])
    fs.write("validation_passed",int(passed));fs.release()
    fs=cv2.FileStorage(str(out/"intrinsics_used.xml"),cv2.FILE_STORAGE_WRITE)
    fs.write("K",np.array(camera["K"]).reshape(3,3))
    fs.write("distortion",np.array(camera["D"]).reshape(1,-1))
    fs.write("width",camera["width"]);fs.write("height",camera["height"])
    fs.write("distortion_model",camera["distortion_model"])
    fs.write("source","capture manifest; SDK provenance not independently verified");fs.release()
    xyz = " ".join(f"{a/1000.:.12f}" for a in X[:3,3])
    rpy = " ".join(f"{a:.12f}" for a in angles_rad)
    (out/"head_rgb_optical.urdf.xml").write_text(f'''<!-- Reference snippet only; not deployed.
     RGB optical frame: X right, Y down, Z forward.
     Uses original measured q_ours without extra joint-zero corrections.
     Status: {status}; this is not an all-points <=5mm guarantee.
     Do not paste this optical transform into a physical camera_link joint. -->
<link name="calibrated_rgb_optical_frame"/>
<joint name="calibrated_rgb_optical_joint" type="fixed">
  <parent link="head_pitch_Link"/>
  <child link="calibrated_rgb_optical_frame"/>
  <origin xyz="{xyz}" rpy="{rpy}"/>
</joint>
''')
    short_names = ["PARK 初值", "联合重投影（初始化阶段）"]
    if report["primary_objective"] == "fullboard":
        short_names.append("全板三维优化（名义关节）")
    short_names.append("相机外参＋棋盘位姿＋零偏联合优化（最终）")
    table = []
    for name, result in zip(short_names, report["results"]):
        tr, va = result["train"], result["validation"]
        table.append(f"| {name} | {tr['origin_mm']['rms']:.4f} | {va['origin_mm']['rms']:.4f} | {va['origin_mm']['max']:.4f} | {va['all_corners_to_Y_mm']['rms']:.4f} |")
    rank = report["joint_offset_identifiability"]
    matrix = "\n".join(" ".join(f"{a: .10f}" for a in row) for row in X)
    total = len(detection)
    rang = np.array(report["joint_ranges_deg"])
    date = manifest["created"].split("T")[0]
    relative_output = out.as_posix()
    conclusion = "达到本批留出验证的 RMS ≤5 mm 要求" if passed else "未达到本批留出验证的 RMS ≤5 mm 要求"
    memo = f'''# {date} PARK 初始化与关节零偏/相机外参联合标定

## 结论与适用范围

**{conclusion}。** 最终板原点三维 RMS **{v['origin_mm']['rms']:.4f} mm**，最大 **{v['origin_mm']['max']:.4f} mm**；全部角点三维 RMS **{v['all_corners_to_Y_mm']['rms']:.4f} mm**，最大 **{v['all_corners_to_Y_mm']['max']:.4f} mm**。

{total}张图像检测成功{len(found)}张。按原始pose_index排序，前{training_end}个姿态训练，后{report['validation_count_requested']}个姿态留出验证，分组在角点检测与拟合前保存。没有根据残差删除图像、改变棋盘尺度或重分验证集。

这是**同一固定板采集session内、未参与求解的姿态一致性验证**，不是外部真值绝对精度测试。本批棋盘的PnP光轴Z范围约 **{depths.min()/1000.:.3f}～{depths.max()/1000.:.3f} m**；不能据此宣布此前约1～1.2m工作距离、深度点云或机械臂抓取精度也已通过。

验证原点超过5mm的姿态序号为：{exceeded}。验收条件是用户本次指定的RMS≤5mm，**不是所有点最大误差≤5mm**。

## 预先固定的算法与数据

- 数据目录：`{capture.name}`。使用其中manifest与原始图像，不修改源数据。
- 棋盘：{board['cols']}×{board['rows']}内角点（12×9方格），格距{board['square_mm']:g}mm。沿用此前用户确认值；图中标识也为15mm规格，没有独立尺寸计量。
- RGB：{camera['width']}×{camera['height']}，固定manifest的K、D与畸变模型，不在此次求解中重标定内参。
- K：fx={camera['K'][0]:.9f}、fy={camera['K'][4]:.9f}、cx={camera['K'][2]:.9f}、cy={camera['K'][5]:.9f}。
- D={camera['D']}，模型{camera['distortion_model']}。真实SDK畸变来源仍未独立核实；当前发布器写死零畸变的情况需要与采集端确认。
- 机器人位姿来自实测`q_ours`；逐条核对其与`q_raw × hardware_direction`一致，未采用commanded角替代。
- FK沿用`tools/reference_urdf/shuangbi20260803.urdf`的yaw/pitch轴线和原点，基座`waist_yaw_Link`、末端`head_pitch_Link`。
- yaw范围：{rang[0,0]:.4f}°～{rang[1,0]:.4f}°；pitch范围：{rang[0,1]:.4f}°～{rang[1,1]:.4f}°。manifest里的旧范围提示不作为实测范围。
- 检测：SB后半窗口{report['corner_refinement_half_window']}px的角点精修；本批各帧角点间距中位数范围{spacing.min():.2f}～{spacing.max():.2f}px，总体中位数{np.median(spacing):.2f}px。
- 单帧独立PnP像素RMS范围{per_view_px.min():.4f}～{per_view_px.max():.4f}px。它与闭环手眼重投影误差不是同一指标。
- 方法在观察新数据验证结果前确定：PARK → Huber(1px)联合重投影初始优化 → 全板三维最小二乘 → 加入两个全局关节零偏及零偏规范约束。
- 相机位姿、棋盘位姿和两个全局零偏均只由训练数据及明确的零偏先验求解。没有每帧姿态补偿、验证中心重估或验证误差调参。
- 重投影初始阶段原点RMS可能比最终阶段更小，但本次按预先指定的全板三维目标导出最终矩阵，没有根据验证成绩另选一个矩阵。

## 结果对比

| 方法 | 训练原点 RMS mm | 验证原点 RMS mm | 验证原点 Max mm | 验证全板角点 RMS mm |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

板原点误差为：每张图独立PnP的板原点，经该帧FK和候选手眼矩阵变换后，到**训练组变换后板原点均值**的三维距离。验证图像仅用于评价，不影响这个均值。

若改以训练联合求得的固定板位姿Y为参考，验证原点RMS为{v['origin_to_optimized_Y_mm']['rms']:.4f}mm。全板三维误差以训练Y作为88个角点的参考；1760个验证角点来自20个姿态，不能把它们当成1760个独立姿态。

验证XYZ分量RMS为{v['axis_rms_mm'][0]:.4f}、{v['axis_rms_mm'][1]:.4f}、{v['axis_rms_mm'][2]:.4f}mm。闭环重投影RMS为{v['reprojection_px']['rms']:.4f}px；这个值未被隐藏，三维目标与像素目标不同，不宣称所有相机模型误差已经消除。

## 关节零偏的可辨识性

只有两个串联头部关节，固定棋盘的基座位姿Y和相机外参X都未知。令：

```text
G(q) = O_yaw R_yaw(q_yaw) O_pitch R_pitch(q_pitch)
G(q + delta) X C_i = Y
B = O_yaw R_yaw(delta_yaw) inverse(O_yaw)
X_effective = R_pitch(delta_pitch) X
Y_effective = inverse(B) Y
则 G(q) X_effective C_i = Y_effective
```

yaw零偏可被棋盘/基座关系吸收，pitch零偏可被相机安装外参吸收。因此本批数据**不能独立测出两个真实物理零偏**。

14个参数的无先验图像观测雅可比秩为{rank['rank_relative_tolerance_1e_7']}/14；两个自由度未被观测。先验尺度为1°、中心为0，仅用于确定等价解的坐标约定。未经换算的优化偏置为yaw={final['offsets_yaw_pitch_deg'][0]:.9f}°、pitch={final['offsets_yaw_pitch_deg'][1]:.9f}°；这些不是可信物理零位测量结果。

导出的X已换算为原始q_ours下的**等效外参**，应继续使用原始q_ours，额外零偏补偿设为0，不能再重复叠加上述偏置。规范换算前后投影最大差约{rank['canonical_projection_difference_max_px']:.3e}px。若要标定真实物理零偏，需要另外提供足以分开首/末关节零位与X/Y的外部几何或零位参考。

## 最终外参与使用约定

矩阵方向是RGB光学坐标系到head_pitch_Link，平移单位mm：

```text
p_head_pitch_Link = R * p_RGB_optical + t

{matrix}
```

- 平移XYZ(mm)：({X[0,3]:.6f}, {X[1,3]:.6f}, {X[2,3]:.6f})。
- 光学系外参RPY(deg，固定轴XYZ)：({np.degrees(angles_rad)[0]:.6f}, {np.degrees(angles_rad)[1]:.6f}, {np.degrees(angles_rad)[2]:.6f})。
- 原始图像使用本批对应的K、D。目标点进入X之前必须已经在RGB optical坐标系中，长度单位必须一致。
- 变换到基座：`p_base = G(q_ours) X p_RGB_optical`。
- URDF参考片段使用m和rad，并明确创建RGB optical子坐标系。不能把光学系的RPY直接贴入实体camera_link安装关节；实体link换算需要实际link到RGB optical的完整固定变换。
- 本次没有写入机器人URDF、关节零位或运行中的TF，也没有用全部100姿态重拟合后沿用旧验证分数。

## 验收边界与后续验证

已完成本次指定的留出姿态RMS评价。若应用要求最大误差也≤5mm，或工作距离为1～1.2m，仍需按该工况另采验证数据。绝对定位需要外部已知坐标参考；RGB/PnP一致性不能替代深度点云、机械臂基座链或TCP验收。

本批每姿态一张图，缺少独立曝光/PDO时间戳对，无法仅凭manifest审核图像与关节的精确同步。棋盘15mm尺寸、真实SDK畸变以及URDF与实物的一致性仍是使用结果的前提。

## 文件与复跑

| 文件 | 内容 |
|---|---|
| [calibration_result.json](calibration_result.json) | 最终矩阵、指标、坐标约定、适用范围与验收状态 |
| [calib_extrinsic.xml](calib_extrinsic.xml) | OpenCV R/t与4×4矩阵；t是3×1、单位mm |
| [intrinsics_used.xml](intrinsics_used.xml) | 本批固定使用的K、D、尺寸 |
| [head_rgb_optical.urdf.xml](head_rgb_optical.urdf.xml) | RGB optical固定关节参考，未部署 |
| [joint_per_frame.csv](joint_per_frame.csv) | 最终全部100帧误差与分组 |
| [park_per_frame.csv](park_per_frame.csv) | PARK全部100帧误差 |
| [results.json](results.json) | 所有预定求解阶段、优化状态、可辨识性检查 |
| [split.json](split.json) | 预先确定的分组和算法配置 |
| [detection.json](detection.json) | 检测、像素间距、单图PnP统计 |
| [error_comparison.png](error_comparison.png) | 误差对比图 |
| `corner_cache_metadata.json` | 输入图像、manifest哈希及检测缓存配置 |
| `source_snapshot/` | 本次所用脚本副本，便于后续追溯 |

在仓库根目录复跑（系统Python禁用用户site-packages）：

```bash
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -s tools/calibrate_head_joint.py \\
  --capture {capture.name} --output {relative_output} \\
  --validate-count {report['validation_count_requested']} --corner-refinement {report['corner_refinement_half_window']} --objective {report['primary_objective']}
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -s tools/export_head_calibration.py \\
  --output {relative_output}
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -s tools/test_calibrate_head_joint.py
```

求解版本：OpenCV {report['versions']['opencv']}、NumPy {report['versions']['numpy']}、SciPy {report['versions']['scipy']}。数值测试覆盖独立URDF FK、PARK理想真值恢复、零偏等价变换及含噪三维目标不变性、验证数据不影响训练参考中心。图像和manifest按哈希核对，原始数据保持不变。
'''
    (out/"REPORT.md").write_text(memo)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    for item,label in ((report["results"][0],"PARK"),(report["results"][1],"Reprojection initialization"),(final,"Final full-board + joint offsets")):
        rows=item["per_frame"]
        axes[0].plot([r["index"] for r in rows],[r["origin_error_mm"] for r in rows],".-",markersize=3,linewidth=.8,label=label)
        rows=[r for r in rows if r["split"]=="validation"]
        axes[1].plot([r["index"] for r in rows],[r["origin_error_mm"] for r in rows],"o-",label=label)
    axes[0].axvspan(training_end-.5,total-.5,color="grey",alpha=.12,label="Holdout")
    for ax in axes:
        ax.set_xlabel("Original pose index");ax.set_ylabel("3D board-origin error (mm)")
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_title(f"{date}: {training_end} training / {report['validation_count_requested']} holdout; no images removed")
    axes[1].set_title(f"Final holdout RMS {v['origin_mm']['rms']:.3f} mm; max {v['origin_mm']['max']:.3f} mm")
    axes[1].set_xticks([f["index"] for f in validation])
    fig.savefig(out/"error_comparison.png",dpi=160);plt.close(fig)
    snapshot=out/"source_snapshot";snapshot.mkdir(exist_ok=True)
    for name in ("calibrate_head_joint.py","manifest_to_dataset.py","test_calibrate_head_joint.py","export_head_calibration.py"):
        shutil.copyfile(Path(__file__).parent/name,snapshot/name)
    outputs=["calibration_result.json","calib_extrinsic.xml","intrinsics_used.xml",
             "head_rgb_optical.urdf.xml","results.json","joint_per_frame.csv","REPORT.md"]
    hashes={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in outputs}
    (out/"output_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(status, "exported to",out.resolve())


if __name__ == "__main__":
    main()
