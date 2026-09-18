# 2026-09-18 PARK 初始化与关节零偏/相机外参联合标定

## 结论与适用范围

**达到本批留出验证的 RMS ≤5 mm 要求。** 最终板原点三维 RMS **3.4242 mm**，最大 **5.2376 mm**；全部角点三维 RMS **3.1532 mm**，最大 **5.4498 mm**。

100张图像检测成功100张。按原始pose_index排序，前80个姿态训练，后20个姿态留出验证，分组在角点检测与拟合前保存。没有根据残差删除图像、改变棋盘尺度或重分验证集。

这是**同一固定板采集session内、未参与求解的姿态一致性验证**，不是外部真值绝对精度测试。本批棋盘的PnP光轴Z范围约 **0.485～0.570 m**；不能据此宣布此前约1～1.2m工作距离、深度点云或机械臂抓取精度也已通过。

验证原点超过5mm的姿态序号为：[83, 91]。验收条件是用户本次指定的RMS≤5mm，**不是所有点最大误差≤5mm**。

## 预先固定的算法与数据

- 数据目录：`handeye_2026-09-18`。使用其中manifest与原始图像，不修改源数据。
- 棋盘：11×8内角点（12×9方格），格距15mm。沿用此前用户确认值；图中标识也为15mm规格，没有独立尺寸计量。
- RGB：1280×720，固定manifest的K、D与畸变模型，不在此次求解中重标定内参。
- K：fx=910.959594727、fy=908.497863770、cx=636.727172852、cy=380.101501465。
- D=[0.0, 0.0, 0.0, 0.0, 0.0]，模型plumb_bob。真实SDK畸变来源仍未独立核实；当前发布器写死零畸变的情况需要与采集端确认。
- 机器人位姿来自实测`q_ours`；逐条核对其与`q_raw × hardware_direction`一致，未采用commanded角替代。
- FK沿用`tools/reference_urdf/shuangbi20260803.urdf`的yaw/pitch轴线和原点，基座`waist_yaw_Link`、末端`head_pitch_Link`。
- yaw范围：-18.4818°～18.4007°；pitch范围：-1.7578°～16.0085°。manifest里的旧范围提示不作为实测范围。
- 检测：SB后半窗口4px的角点精修；本批各帧角点间距中位数范围24.13～27.57px，总体中位数25.04px。
- 单帧独立PnP像素RMS范围0.1205～0.5176px。它与闭环手眼重投影误差不是同一指标。
- 方法在观察新数据验证结果前确定：PARK → Huber(1px)联合重投影初始优化 → 全板三维最小二乘 → 加入两个全局关节零偏及零偏规范约束。
- 相机位姿、棋盘位姿和两个全局零偏均只由训练数据及明确的零偏先验求解。没有每帧姿态补偿、验证中心重估或验证误差调参。
- 重投影初始阶段原点RMS可能比最终阶段更小，但本次按预先指定的全板三维目标导出最终矩阵，没有根据验证成绩另选一个矩阵。

## 结果对比

| 方法 | 训练原点 RMS mm | 验证原点 RMS mm | 验证原点 Max mm | 验证全板角点 RMS mm |
|---|---:|---:|---:|---:|
| PARK 初值 | 7.9279 | 7.6371 | 13.1290 | 7.5009 |
| 联合重投影（初始化阶段） | 3.4806 | 3.2642 | 5.5914 | 3.4544 |
| 全板三维优化（名义关节） | 3.5826 | 3.4242 | 5.2376 | 3.1532 |
| 相机外参＋棋盘位姿＋零偏联合优化（最终） | 3.5826 | 3.4242 | 5.2376 | 3.1532 |

板原点误差为：每张图独立PnP的板原点，经该帧FK和候选手眼矩阵变换后，到**训练组变换后板原点均值**的三维距离。验证图像仅用于评价，不影响这个均值。

若改以训练联合求得的固定板位姿Y为参考，验证原点RMS为3.4234mm。全板三维误差以训练Y作为88个角点的参考；1760个验证角点来自20个姿态，不能把它们当成1760个独立姿态。

验证XYZ分量RMS为3.2198、0.9528、0.6709mm。闭环重投影RMS为2.0548px；这个值未被隐藏，三维目标与像素目标不同，不宣称所有相机模型误差已经消除。

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

14个参数的无先验图像观测雅可比秩为12/14；两个自由度未被观测。先验尺度为1°、中心为0，仅用于确定等价解的坐标约定。未经换算的优化偏置为yaw=0.000008097°、pitch=-0.000053355°；这些不是可信物理零位测量结果。

导出的X已换算为原始q_ours下的**等效外参**，应继续使用原始q_ours，额外零偏补偿设为0，不能再重复叠加上述偏置。规范换算前后投影最大差约9.095e-13px。若要标定真实物理零偏，需要另外提供足以分开首/末关节零位与X/Y的外部几何或零位参考。

## 最终外参与使用约定

矩阵方向是RGB光学坐标系到head_pitch_Link，平移单位mm：

```text
p_head_pitch_Link = R * p_RGB_optical + t

 0.0097186472  0.0318988815  0.9994438500  48.2865837444
-0.9999527728  0.0003161121  0.0097135068  71.5464488356
-0.0000060863 -0.9994910512  0.0319004472 -27.7481979810
 0.0000000000  0.0000000000  0.0000000000  1.0000000000
```

- 平移XYZ(mm)：(48.286584, 71.546449, -27.748198)。
- 光学系外参RPY(deg，固定轴XYZ)：(-88.171929, 0.000349, -89.443154)。
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
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -s tools/calibrate_head_joint.py \
  --capture handeye_2026-09-18 --output calibration_output_20260918_replay \
  --validate-count 20 --corner-refinement 4 --objective fullboard
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -s tools/export_head_calibration.py \
  --output calibration_output_20260918_replay
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -s tools/test_calibrate_head_joint.py
```

求解版本：OpenCV 4.5.4、NumPy 1.21.5、SciPy 1.8.0。数值测试覆盖独立URDF FK、PARK理想真值恢复、零偏等价变换及含噪三维目标不变性、验证数据不影响训练参考中心。图像和manifest按哈希核对，原始数据保持不变。
