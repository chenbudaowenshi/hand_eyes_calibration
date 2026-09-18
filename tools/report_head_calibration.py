#!/usr/bin/env python3
"""Produce the auditable report for the 2026-09-17 calibration experiment."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    main = json.loads((out/"results.json").read_text())
    extra = json.loads((out/"additional_diagnostics.json").read_text())
    detections = json.loads((out/"detection.json").read_text())
    chosen = next(r for r in extra["results"] if r["method"] == "SB_subpix4 full-board 3D least squares")
    X = np.array(chosen["X_nominal_head_camera_optical_mm"])
    candidate = {"status": "NOT_PASSED_5MM", "deployed": False,
                 "transform": "p_head_pitch_Link = R * p_RGB_optical + t",
                 "translation_unit": "mm", "joint_state_convention": "original q_ours, no physical zero correction",
                 "joint_offsets_identifiable": False,
                 "X_head_pitch_RGB_optical": X.tolist(),
                 "rpy_optical_fixed_xyz_deg": Rotation.from_matrix(X[:3,:3]).as_euler("xyz",degrees=True).tolist(),
                 "validation": chosen["validation"],
                 "method": chosen["method"],
                 "note": "Exploratory follow-up after initial holdout; requires fresh validation if adopted."}
    (out/"candidate_fullboard.json").write_text(json.dumps(candidate,ensure_ascii=False,indent=2)+"\n")
    fs=cv2.FileStorage(str(out/"candidate_fullboard_extrinsic.xml"),cv2.FILE_STORAGE_WRITE)
    fs.write("R",X[:3,:3]);fs.write("t",X[:3,3]);fs.write("T_head_camera_optical",X)
    fs.write("translation_unit","mm");fs.write("validation_passed",0)
    fs.write("validation_rms_mm",chosen["validation"]["origin_mm"]["rms"])
    fs.write("note","NOT PASSED 5 mm. Effective extrinsic for nominal q_ours. Not deployed.")
    fs.release()
    with (out/"fullboard_per_frame.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=chosen["per_frame"][0].keys())
        writer.writeheader();writer.writerows(chosen["per_frame"])
    # Summaries do not replace or hide any of the intermediate runs.
    selected=[("PARK",main["results"][0]),
              ("PARK + 联合重投影 + 零偏规范约束",main["results"][-1]),
              ("SB 精修 + 联合重投影",extra["results"][3]),
              ("SB 精修 + 全板三维最小二乘",chosen),
              ("SB 精修 + 仅原点三维最小二乘（诊断）",extra["results"][-1])]
    lines=[]
    for name,r in selected:
        tr,va=r["train"]["origin_mm"],r["validation"]["origin_mm"]
        lines.append(f"| {name} | {tr['rms']:.3f} | {va['rms']:.3f} | {va['max']:.3f} | {r['validation']['all_corners_to_Y_mm']['rms']:.3f} |")
    matrix="\n".join(" ".join(f"{v: .9f}" for v in row) for row in X)
    spacing=np.array([d["spacing_median_px"] for d in detections if d["found"]])
    z=np.array([d["pnp_z_mm"] for d in detections if d["found"]])
    rank=main["joint_offset_identifiability"]
    text=f"""# 2026-09-17 手眼标定与关节零偏联合优化报告

## 结论

**本次未达到验证 RMS ≤5 mm，未部署标定结果。** 100 张全部成功检测；固定前 80 个姿态（pose_00～79）训练、后 20 个姿态（pose_80～99）验证。分组在角点检测和拟合前保存，未删除验证图或根据误差重分组。

主流程 PARK 初始化 + 相机/棋盘位姿与两个关节零偏联合重投影优化，验证原点三维 RMS 为 **{main['results'][-1]['validation']['origin_mm']['rms']:.3f} mm**。

后续角点精修与全板三维优化的验证原点 RMS 为 **{chosen['validation']['origin_mm']['rms']:.3f} mm**、最大 **{chosen['validation']['origin_mm']['max']:.3f} mm**，全板角点 RMS 为 **{chosen['validation']['all_corners_to_Y_mm']['rms']:.3f} mm**。只优化板原点的诊断结果 RMS 为 {extra['results'][-1]['validation']['origin_mm']['rms']:.3f} mm，也没有达到目标。

这些毫米误差均为固定板一致性，不是相对于外部真值的绝对定位误差。当前验证是同一 session 留出姿态，不是另一次采集的完整独立 session。

## 数据与固定模型

- 输入：`handeye_2026-09-17/manifest.json` 及其 `images/`，100 张原始 BGR 图像。
- 检测配置：11×8 内角点、15 mm 格距；格距沿用用户给定值，未为降低误差改变尺度。
- 原始尺寸：1280×720。fx=910.959595、fy=908.497864、cx=636.727173、cy=380.101501。
- 畸变：使用 manifest 声明的 plumb_bob、D=[0,0,0,0,0]，未重新拟合内参。无法仅凭 manifest 证明零畸变来自实际 SDK；当前仓库发布器写死 D=0 是待核实项。
- 机器人：使用 `q_ours` 实测角，逐条校验 q_raw×hardware_direction 一致；没有改用 commanded 角。
- FK：`tools/reference_urdf/shuangbi20260803.urdf`，`waist_yaw_Link -> head_yaw_joint -> head_pitch_joint -> head_pitch_Link`，使用 URDF 原始轴线和原点。
- yaw 范围约 -9.59°～34.88°，pitch 范围约 -7.11°～12.44°。manifest 的“pitch 仅13°”文字不能作为本批实测范围。
- 棋盘距离 PnP Z 约 {z.min():.1f}～{z.max():.1f} mm；各帧相邻角点间距中位数范围 {spacing.min():.2f}～{spacing.max():.2f} px，总体中位数 {np.median(spacing):.2f} px。仍低于此前争取的20～30px采集设计目标。
- 图像 SHA256、manifest/URDF 哈希、分组及逐帧检测指标均保存在输出目录中。

## 为什么不能独立标定两个真实关节零偏

设头部运动链为：

```text
G(q) = O_yaw R_yaw(q_yaw) O_pitch R_pitch(q_pitch)
G(q + delta) X C_i = Y
```

其中 X 为相机到末端外参，Y 为未知的固定板到基座位姿。存在以下严格等价变换：

```text
B = O_yaw R_yaw(delta_yaw) inverse(O_yaw)
X_effective = R_pitch(delta_pitch) X
Y_effective = inverse(B) Y
G(q) X_effective C_i = Y_effective
```

因此首关节 yaw 零偏可被未知板位姿吸收，末关节 pitch 零偏可被相机外参吸收。此处缺少的是能分开这些参数的外部参考；增加同类图像不能消除这两个自由度。

本次14参数包括相机6、棋盘6、零偏2。数值数据雅可比秩为 **{rank['rank_relative_tolerance_1e_7']}/14**；末尾两个奇异值接近零。将联合解换算回名义零偏约定后，投影最大变化约 {rank['canonical_projection_difference_max_px']:.2e} px，验证了上述等价性。

求解中加入零偏以0为中心、尺度1°的先验，只用于选定一个等价坐标约定。输出的接近0零偏是该约束的结果，**不能解释为测得真实零偏为0，也不能写入电机零位**。两个偏置一起自由拟合得到其他数值，也不代表精度更高。

输出候选外参均换算为原始 `q_ours` 的等效外参，使用时不应再叠加一份零偏补偿。要分别求真实零偏和安装外参，需要可靠外部约束，例如已测量的板到基座关系、独立关节零位参考，以及能约束末关节与相机安装关系的测量；约束的误差也须计入。

## 求解与评价

原图只使用一套 K/D；未去畸变后重复应用 D。主流程使用 SB 亚像素检测、逐图 PnP、训练集 PARK 初值，然后以 Huber(1px) 优化全部训练角点的重投影。优化变量只有固定的 X、Y、两个全局零偏，没有每帧姿态修正。

追加诊断对 SB 角点使用半窗口4×4、实际9×9支持域的 cornerSubPix；支持域小于本批角点间距。比较重投影、全板三维、仅板原点三维三种目标。每种模型仍只拟合前80个姿态，全部试验结果保留。

**追加诊断发生在首次查看留出结果之后，因此不能将其当作完全未用于开发的最终盲测。** 若将其中的新处理方式用于正式验收，应重新采集确认集。本次所有方法都未达到5mm，不存在据此宣称通过的问题。

位置残差使用同一口径：每帧独立 PnP 所得板原点经过 G_i X 变换后，到仅由训练组计算的板原点均值的三维距离。验证组不重新估计中心。另报告到训练优化所得 Y 的原点与全部88角点误差，及像素误差。

| 方法 | 训练原点 RMS mm | 验证原点 RMS mm | 验证原点 Max mm | 验证全板角点 RMS mm |
|---|---:|---:|---:|---:|
{chr(10).join(lines)}

全板三维候选的验证 XYZ 分量 RMS 分别为 {chosen['validation']['axis_rms_mm'][0]:.3f}、{chosen['validation']['axis_rms_mm'][1]:.3f}、{chosen['validation']['axis_rms_mm'][2]:.3f} mm，主要误差沿基座X方向，该方向在本场景大致对应到棋盘的距离方向。这提示应重点核查深度/尺度相关视觉误差，但不能据此断定唯一根因。

## 未达标候选外参

保存全板三维候选用于复核，不替换正在使用的标定。矩阵为：

```text
p_head_pitch_Link = R * p_RGB_optical + t
t 的单位：mm；对应原始 q_ours，关节零偏约定为0。

{matrix}
```

平移约 ({X[0,3]:.3f}, {X[1,3]:.3f}, {X[2,3]:.3f}) mm。

这是 RGB 光学坐标系外参，不能直接当作实体 camera_link 安装变换。若需要实体 link 外参，必须组合实际 link↔RGB optical 的完整固定变换，包括可能存在的平移。

## 后续优先事项

1. **核实实际 SDK 畸变模型和系数。** 当前 D=0 来自 manifest，来源链不足以确认真实光学畸变。不要为得到5mm而在此批小棋盘图上任意放开全部内参。
2. **提高实际角点采样密度。** 本批每格仍约13～16px；保留15mm棋盘和1280×720时可在允许的标定距离内靠近，或使用更大板。必须在实际工作距离另行验证，不能拿近距离成绩代替远距离要求。
3. **同姿态连续图像与曝光/关节时间配对。** 当前每姿态只有一张，缺少独立的曝光与PDO时间戳，无法从已有文件区分检测/PnP抖动、同步误差和结构误差。指令跟踪差值不是编码器测量精度，不能直接用它当每帧姿态补偿。
4. **若必须标定物理零偏，补充外部参考。** 两轴端点零偏与外参的不可分离性是模型约束，不是换优化器就能解决的问题。
5. **冻结新流程后另采约20～25个验证姿态。** 当前结果不能宣布5mm通过，也没有证据保证仅靠上述某一项就必然达标。

## 文件与复跑

| 文件 | 用途 |
|---|---|
| `results.json` | PARK与主流程联合优化、秩检查、验收状态 |
| `additional_diagnostics.json` | 全部追加诊断方法结果，没有隐藏失败试验 |
| `split.json` | 预先固定的80/20分组和输入哈希 |
| `detection.json` | 100张图的检测、像素间距、PnP误差 |
| `corner_cache_metadata.json` | 每张图SHA256及缓存来源 |
| `park_per_frame.csv`、`joint_per_frame.csv` | PARK和主流程逐帧误差 |
| `fullboard_per_frame.csv` | 全板三维候选逐帧误差 |
| `candidate_extrinsic.xml` | 主流程未达标候选，附validation_passed=0 |
| `candidate_fullboard.json`、`candidate_fullboard_extrinsic.xml` | 全板三维未达标候选，明确状态 |
| `error_comparison.png` | 全部姿态误差与固定20帧验证误差图 |
| `corners.npz`、`observations.npz`、`refined_corners.npz` | 检测与求解中间数据 |

在仓库根目录运行，使用系统Python禁用用户site-packages，避免用户NumPy与系统SciPy版本范围不匹配：

```bash
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -s tools/calibrate_head_joint.py \\
  --capture handeye_2026-09-17 --output calibration_output_20260917_joint --validate-count 20
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -s tools/refine_head_diagnostics.py \\
  --capture handeye_2026-09-17 --output calibration_output_20260917_joint
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -s tools/report_head_calibration.py \\
  --output calibration_output_20260917_joint
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -s tools/test_calibrate_head_joint.py
```

已使用 OpenCV {main['versions']['opencv']}、NumPy {main['versions']['numpy']}、SciPy {main['versions']['scipy']} 复跑。四项数值测试覆盖URDF FK、零偏等价变换、无噪声PARK真值恢复、验证数据不能移动训练参考中心。没有修改原始图像、manifest、机器人URDF或实机参数。
"""
    (out/"REPORT.md").write_text(text)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    for r,name in ((main["results"][0],"PARK"),(main["results"][-1],"Joint reprojection"),(chosen,"Refined + full-board 3D")):
        frame=r["per_frame"]
        x=[f["index"] for f in frame];y=[f["origin_error_mm"] for f in frame]
        axs[0].plot(x,y,".-",markersize=3,linewidth=.8,label=name)
        val=[f for f in frame if f["split"]=="validation"]
        axs[1].plot([f["index"] for f in val],[f["origin_error_mm"] for f in val],"o-",label=name)
    axs[0].axvspan(79.5,99.5,color="grey",alpha=.12,label="Held-out poses")
    for ax in axs:
        ax.set_ylabel("3D board-origin error (mm)");ax.set_xlabel("Original pose index")
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    axs[0].set_title("Training: poses 00-79; validation: poses 80-99; no frames removed")
    axs[1].set_title("Held-out pose errors (5 mm target applies to RMS, not individual error)")
    axs[1].set_xticks(range(80,100))
    fig.savefig(out/"error_comparison.png",dpi=160);plt.close(fig)
    print("Saved",out/"REPORT.md")


if __name__ == "__main__":
    main()
