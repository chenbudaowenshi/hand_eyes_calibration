# 头部 D435i 手眼标定：运行方法、问题分析与逐步优化记录

更新日期：2026-09-18。

本文依据当前仓库代码、三批实际采集数据和已保存的求解报告编写，说明程序怎么运行、误差为什么较大、采用了哪些改进，以及当前结果可以用于什么范围。

## 1. 当前结果与项目范围

最新数据 `handeye_2026-09-18` 的100张图像全部检测成功。固定前80个姿态训练、后20个姿态验证，按预先确定的流程运行：

```text
SB角点检测及精修
    → 固定内参PnP
    → PARK初始化
    → 联合重投影优化
    → 全板三维优化
    → 相机外参、固定板位姿与关节零偏联合优化
    → 换算为原始关节约定下的等效外参
    → 留出姿态验证与文件导出
```

| 方法 | 训练原点 RMS / mm | 验证原点 RMS / mm | 验证原点 Max / mm | 验证全板角点 RMS / mm |
|---|---:|---:|---:|---:|
| PARK初值 | 7.9279 | 7.6371 | 13.1290 | 7.5009 |
| 联合重投影初始化 | 3.4806 | 3.2642 | 5.5914 | 3.4544 |
| 全板三维优化 | 3.5826 | 3.4242 | 5.2377 | 3.1532 |
| 最终相机/棋盘/零偏联合优化 | 3.5826 | **3.4242** | **5.2377** | **3.1532** |

本次达到了用户要求的**验证RMS ≤5 mm**。需要准确理解这个结论：

- 验证原点最大误差为5.2377 mm，姿态83、91超过5 mm，因此尚未达到“所有验证点最大误差≤5 mm”。
- 棋盘光轴方向距离约0.485～0.570 m；没有据此验证1～1.2 m范围。
- 这是同一采集批次内、未参与拟合的姿态对固定棋盘的三维一致性，不是相对于外部计量真值的绝对定位精度。
- RGB/PnP的结果不能直接替代深度点云、机械臂基座关系、TCP或实际抓取精度。
- 关节零偏与相机/棋盘位姿之间存在两个不可分离的自由度。最终是原始`q_ours`约定下的等效外参，不能宣称已经独立测出了两个物理关节零偏。

完整原始结果见[2026-09-18标定报告](calibration_output_20260918_joint/REPORT.md)。本次没有修改实机URDF、关节零位或运行中的TF。

## 2. 程序运行方法

### 2.1 区分采集程序、相机发布器与离线标定程序

| 入口 | 用途 | 本次是否需要 |
|---|---|---|
| 外部采集程序 | 保存图片、实测关节状态和manifest | 已采完时不需要再运行 |
| `src/camera/run_realsense_viewer.py` | RealSense预览或ROS图像发布 | 现场检查或供采集端使用 |
| `tools/calibrate_head_joint.py` | 本次使用的PARK及联合优化求解 | 需要 |
| `tools/export_head_calibration.py` | 导出最终JSON、XML、报告、误差图与URDF参考片段 | 需要 |
| `tools/test_calibrate_head_joint.py` | 不连接硬件的数值正确性测试 | 建议在改动代码后运行 |
| `tools/refine_head_diagnostics.py` | 9月17日开发时的追加诊断与目标函数对比 | 正常复跑9月18日结果不需要 |
| `tools/report_head_calibration.py` | 9月17日专用的历史诊断报告生成器 | 不用于9月18日正式导出 |
| `./build/calib` | 原有C++标定入口 | 不是此次完整联合优化流程的入口 |

实际采集程序不在`src/camera`中。不要把启动预览器理解为已经自动保存了机器人手眼数据，也不要把旧C++程序的结果当成新Python流程的结果。

### 2.2 运行目录与Python环境

离线标定无需连接相机，也不要求ROS正在运行。

```bash
cd /home/robot/hand_eyes_calibration

/usr/bin/python3 -s - <<'PY'
import cv2
import numpy
import scipy
import matplotlib
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("Matplotlib:", matplotlib.__version__)
PY
```

本次求解使用OpenCV 4.5.4、NumPy 1.21.5、SciPy 1.8.0。Matplotlib用于生成报告中的误差图。

采用`/usr/bin/python3 -s`是因为本机用户目录中另一套NumPy曾与系统SciPy的声明支持范围不匹配。`-s`禁用用户site-packages，使本次使用的系统依赖版本保持一致。`OPENBLAS_NUM_THREADS=1`限制BLAS并行度，减少小规模求解的线程开销；这不是提高精度的参数。

如果缺少依赖，应在明确的环境中安装兼容版本，不要混用不同Python环境后只比较最终数值。其他环境的版本变化也应记录。

### 2.3 输入目录与必要字段

`--capture`指定**包含manifest.json的根目录**，不要指向它的`images/`子目录：

```text
handeye_2026-09-18/
├── manifest.json
└── images/
    ├── pose_00.png
    ├── pose_01.png
    └── ... pose_99.png
```

当前程序至少需要以下manifest信息：

| 字段 | 内容 |
|---|---|
| `camera_info.width/height` | 实际图片宽高 |
| `camera_info.K` | 按行展开的9个相机矩阵数值 |
| `camera_info.D` | 与图像、畸变模型匹配的系数 |
| `camera_info.distortion_model` | 当前离线入口接受`plumb_bob`；其他模型需先正确适配 |
| `conventions.hardware_direction` | 每个关节从原始读数到URDF约定的方向映射 |
| `records[].pose_index` | 唯一姿态编号 |
| `records[].image` | 相对于采集根目录的图片路径 |
| `records[].q_raw` | 实测驱动器原始关节角 |
| `records[].q_ours` | 已按方向映射转换的实测关节角，单位rad |

当前机构使用`head_yaw_joint`、`head_pitch_joint`。程序检查`q_ours = q_raw × hardware_direction`，根据URDF计算`waist_yaw_Link → head_pitch_Link`。

生产采集还应保存曝光时间、关节采样时间及各自时钟域、稳定窗口、相机序列号和板尺寸。现有manifest并不包含完整的同步证据。

本次输入是每个姿态一张图片。当前脚本按record排序留出末尾N条，**没有自动按连拍pose_id分组的机制**。未来每姿态保存10～20帧时，需要先按固定规则导出代表帧，或扩展分组逻辑，不能将同姿态连续帧随机分到训练和验证两组。

### 2.4 复现9月18日结果

以下命令使用新的输出目录`calibration_output_20260918_replay`，保留已有验收产物：

```bash
cd /home/robot/hand_eyes_calibration

OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -s tools/calibrate_head_joint.py \
  --capture handeye_2026-09-18 \
  --urdf tools/reference_urdf/shuangbi20260803.urdf \
  --output calibration_output_20260918_replay \
  --validate-count 20 \
  --cols 11 --rows 8 --square-mm 15 \
  --corner-refinement 4 \
  --objective fullboard

MPLCONFIGDIR=/tmp/handeye_mplconfig OPENBLAS_NUM_THREADS=1 \
  /usr/bin/python3 -s tools/export_head_calibration.py \
  --output calibration_output_20260918_replay
```

参数含义：

| 参数 | 本次值 | 说明 |
|---|---|---|
| `--capture` | `handeye_2026-09-18` | 源数据根目录，只读 |
| `--urdf` | 项目reference URDF | 头部轴线、关节原点和关节链 |
| `--output` | 新的结果目录 | 缓存、统计、矩阵和报告输出位置 |
| `--validate-count` | 20 | 原始pose_index排序后末尾20条留出；其余80条训练 |
| `--cols/--rows` | 11/8 | 内角点数量，不是12×9方格数量 |
| `--square-mm` | 15 | 真实格距，不能为了降误差缩小它 |
| `--corner-refinement` | 4 | cornerSubPix半窗口4px，对应9×9px支持域 |
| `--objective` | `fullboard` | 用所有棋盘角点的三维一致性作最终优化目标 |

脚本默认`--corner-refinement 0`、`--objective reprojection`。**要复现本文最终结果，需要显式指定上面的4和fullboard。** 角点窗口不是越大越好，改变棋盘成像尺度后需要检查窗口是否合理。

`MPLCONFIGDIR`只是把Matplotlib缓存放到可写目录，不影响求解结果。输出目录存在时可能覆盖其中的结果文件，因此每个新实验建议采用不同输出目录。

### 2.5 如何判断运行通过

控制台预期最后出现：

```text
RESULT: PASS RMS <= 5 mm; report .../results.json
```

**程序没有报错不等于精度达标。** 当前求解脚本正常完成但精度不通过时，也可能以0退出；必须读取`results.json`里的`goal.passed`，不能只看shell退出码。

可运行下面的检查，将未达标转成非零退出码：

```bash
/usr/bin/python3 -s - <<'PY'
import json
from pathlib import Path

path = Path("calibration_output_20260918_replay/results.json")
data = json.loads(path.read_text())
final = data["results"][-1]
print("PASS:", data["goal"]["passed"])
print("验证姿态数:", final["validation"]["pose_count"])
print("验证原点 RMS/Max(mm):", final["validation"]["origin_mm"])
print("验证全板角点统计(mm):", final["validation"]["all_corners_to_Y_mm"])
raise SystemExit(0 if data["goal"]["passed"] else 1)
PY
```

小数末位可能因依赖和数值实现略有变化，但不应出现明显量级变化。本批预期约为RMS 3.4242 mm、Max 5.2377 mm。

### 2.6 输出文件用途

| 文件 | 用途 |
|---|---|
| `REPORT.md` | 人可阅读的结论、过程、坐标约定与限制 |
| `calibration_result.json` | 最终矩阵、适用范围、原点参考和验收指标 |
| `calib_extrinsic.xml` | OpenCV格式R、t和4×4外参 |
| `intrinsics_used.xml` | 本次固定使用的K、D和尺寸 |
| `head_rgb_optical.urdf.xml` | 光学系固定关节参考片段，未部署 |
| `results.json` | PARK及各优化阶段的完整数值、优化状态、可辨识性检查 |
| `joint_per_frame.csv` | 最终逐帧误差和训练/验证分组 |
| `park_per_frame.csv` | PARK逐帧误差 |
| `split.json` | 在求解前固定的分组和算法配置 |
| `detection.json` | 每图检测、棋盘像素尺寸、单帧PnP质量 |
| `corner_cache_metadata.json` | 图片、manifest哈希和检测缓存参数 |
| `corners.npz`、`observations.npz` | 角点、机器人位姿和求解中间数据 |
| `error_comparison.png` | 全部姿态及留出姿态误差曲线 |
| `source_snapshot/`、`output_hashes.json` | 脚本快照和结果文件哈希 |

第一步求解还输出`candidate_extrinsic.xml`；正式对接优先使用第二步导出的`calib_extrinsic.xml`及配套JSON，前者命名用于候选结果，并不代表已经部署。

### 2.7 相机预览与ROS发布

需要现场检查原图时，可使用本地预览，需要可用的图形显示环境：

```bash
cd /home/robot/hand_eyes_calibration
python3 src/camera/run_realsense_viewer.py \
  --opencv-only --width 1280 --height 720 --camera-fps 30
```

外部采集程序需要ROS图像时：

```bash
cd /home/robot/hand_eyes_calibration
source /opt/ros/humble/setup.bash
python3 src/camera/run_realsense_viewer.py \
  --width 1280 --height 720 --camera-fps 30 --publish-fps 5
```

`--publish-fps 5`表示ROS发布频率，不是相机原始采集帧率。预览/发布器采用能导入`pyrealsense2`和所需ROS包的Python环境；本机RealSense Python包位于用户环境，因此这里没有照搬离线求解的`-s`。

不要同时启动两个程序争用同一台相机。以上命令不自动保存机器人位姿，也不会替代外部采集程序。

## 3. 手眼标定实际求的是什么

本项目相机随头部运动、棋盘相对基座固定，是EIH。manifest中曾写“眼在外”，但它同时说明相机固连头部；求解依据真实运动关系采用EIH。

统一定义`T_A_B`为将B系点转换到A系：

| 符号 | 变换 | 来源 |
|---|---|---|
| G_i | T_base_head | 实测关节角和URDF正运动学 |
| X | T_head_camera_optical | 要求解的手眼外参 |
| C_i | T_camera_optical_board | 原图角点、棋盘尺寸、相机K/D求PnP |
| Y | T_base_board | 训练期间固定但事先未知的棋盘位姿 |

理想闭环关系为：

```text
G_i X C_i = Y
```

工程上遇到的噪声会使每帧算出的Y不完全重合。标定的目标是求一个固定X，使不同姿态的观测尽可能符合这个关系，同时用独立姿态检查泛化表现。

### 3.1 PARK为何适合初始化

PARK使用机器人和棋盘的相对运动求手眼变换。它能给后续非线性求解提供合理初值，但并不直接保证最低的板原点毫米残差或全角点像素误差。

输入方向必须正确：机器人为gripper/head到base；视觉为board/target到camera。两者混用逆矩阵、光学系和实体安装系混淆、米与毫米不一致，都可能导致看似能求解但物理含义错误。

### 3.2 联合重投影优化

固定K、D和真实棋盘点p_j，以机器人观测G_i推导棋盘在相机中的位置：

```text
C_predicted_i = inverse(X) inverse(G_i) Y
像素残差 r_ij = project(K, D, C_predicted_i p_j) - detected_pixel_ij
```

同时优化X和Y，使所有训练角点的投影更接近图像观测。旋转使用旋转向量表示，避免直接优化欧拉角在某些角度附近的表达问题；平移使用mm。

初始化阶段使用Huber损失，像素尺度1 px，使较大的像素残差不会像普通平方损失那样占据过强权重。Huber不会把数据变成零误差，也不等同于删除异常图片。

### 3.3 全板三维优化

为了同时约束棋盘原点和方向，使用全部88个棋盘角点：

```text
三维残差 e_ij = xyz(G_i X C_i p_j) - xyz(Y p_j)
目标 = sum(||e_ij||²)
```

本次该阶段使用普通线性平方损失，不是把Huber同时用于所有阶段。C_i来自固定内参的独立PnP，训练中没有为每张图增加任意位姿修正。

三维目标与像素目标权重不同，因此原点RMS、最大误差、全板RMS和像素RMS不会必然一起下降。9月18日重投影初始化的原点RMS为3.2642 mm，而最终为3.4242 mm；最终全板角点RMS和原点最大误差有所改善。按预先确定的目标导出最终矩阵，没有从验证结果中挑最小的一个数字作为结论。

## 4. 遇到的问题与解决方案

### 4.1 问题：不同算法给出的外参差别很大

9月16日相同数据上，自研、TSAI、PARK、HORAUD、DANIILIDIS给出的结果分歧明显，一些平移达到不合理的米级。

最初将这种分歧直接解释成“姿态信息不足”过于绝对。排查使用真实机器人位姿，合成严格满足闭环的无噪声视觉观测，多个算法都能够恢复设定真值。这说明该运动集合并非天然无法确定手眼变换。

解决思路是分开检查：坐标约定与实现、姿态几何、视觉观测质量、系统误差。PARK/HORAUD一致可以作为诊断线索，不能代替独立精度验收；两者与TSAI都属于先旋转后平移的分离式求解方法，不能错误地按“联合与分离”解释它们的差异。

自研求解器审计还发现旋转行列式校正不足、迭代缺少阻尼/下降保护等实现风险。本次采用OpenCV PARK初始化和SciPy非线性求解路径；没有声称已修复旧自研求解器的所有问题。

### 4.2 问题：照片分辨率提高了，棋盘仍然太小

实际数据的变化如下：

| 批次 | RGB尺寸 | 棋盘距离量级 | 每格成像尺寸量级 |
|---|---|---|---|
| 9月16日 | 640×480 | 约1.1～1.25 m | 约7～8 px |
| 9月17日 | 1280×720 | 约0.90～1.04 m | 约13～16 px |
| 9月18日 | 1280×720 | 约0.49～0.57 m | 约24～28 px |

15 mm方格若只覆盖几个像素，角点微小误差会影响整块平面板的深度与倾角估计。对于近似正视的已知宽度目标，深度量级满足：

```text
Z ≈ fx × 实际宽度 / 成像像素宽度
```

在其他条件相同的近似下，目标成像宽度越小，相同像素扰动造成的相对距离误差越大。这解释了为什么仅提高文件分辨率还不够，要测量棋盘实际在图上的尺寸。

已经做的改进：将两个相机入口默认值改为1280×720；随后9月18日采集在更近距离形成更大的棋盘图像。图像中仍为15 mm规格，没有通过修改物理尺寸来缩小残差。

该解释符合本批误差改善，但三天数据的姿态、距离和图像条件都有变化，不是只改变一个因素的对照实验，不能精确把全部收益归因于分辨率或距离其中一项。

### 4.3 问题：亚像素角点精修窗口过大

旧经典检测分支使用`cornerSubPix(winSize=(11,11))`，这是半窗口，实际覆盖23×23 px。对于每格只有7～8 px的棋盘，支持域可能同时覆盖多个邻近角点。

新离线流程统一使用SB角点检测，并按本次经过检查的像素间距使用半窗口4 px的精修，即9×9 px支持域。

9月17日，在相同分组下：

| 方法 | 验证原点 RMS / mm |
|---|---:|
| SB＋联合重投影 | 7.484 |
| SB＋小窗口精修＋联合重投影 | 6.857 |
| SB＋小窗口精修＋全板三维优化 | 6.446 |

这说明角点处理确实有影响，但只靠窗口调整还不能使那批数据达到5 mm。

本次SB输出按端点位置统一角点方向，适用于当前固定、直立、相机roll较小的棋盘。它不是适用于任意板翻转的身份识别算法；未来大幅roll、板翻转或多块板时，应使用带唯一角点身份的标定板或增加可靠的编号检查。

### 4.4 问题：SDK内参、ROS内参与实际图像可能不一致

两处默认分辨率修改后，相机包装器会从实际RGB profile读取fx、fy和主点。但当前发布器仍把D写成五个零，因此manifest中的零畸变不能独立证明SDK实际系数为零。

当前两次联合标定固定使用各自manifest内参，未凭空修改D，也没有重新拟合内参来降低报告误差。这个前提已经写入验收报告，尚需从采集端核实完整SDK模型和系数。

读取SDK内参不等于重新标定内参。建议流程是先保存实际RGB流的K、D、畸变模型和分辨率，再用约10～15张清晰、多位置/倾角图验证。只有确认模型存在问题、排除了图像处理和棋盘问题后，才考虑独立重标定。

此前在远距离小板数据上自由调整焦距的试验没有改善留出毫米误差，因此没有采用。不能把训练像素误差下降当成内参更准确的唯一证据。

### 4.5 问题：去畸变和PnP重复使用畸变

原C++固定内参路径存在一个潜在问题：先对图像去畸变，后面又把原始D传给PnP。D为零时没有影响，D非零时模型不一致。

新Python求解直接对原图检测，PnP和投影使用对应K、D一次，避免这条处理错误。旧C++路径本身未在此次任务中全面改写，未来切回它并使用非零D时必须修正。

正确的两种处理方式是：

- 原图＋原始K、D。
- 去畸变图＋与该图对应的新K、零D。

不要混用。

### 4.6 问题：实测关节角、指令角、单位和方向混淆

求解使用`q_ours`，不使用`commanded`。当前约定中pitch硬件方向为-1、yaw为+1，程序核对原始读数与转换后读数相符。

用命令角替代实测角，会把未到位误差当成几何误差。反过来，`tracking_err`只是命令与反馈的差值，也不能直接当作编码器测量误差或每帧可自由调整的补偿量。

代码使用URDF原点与轴线进行FK，平移从m转换为mm，再与15 mm棋盘的PnP结果组合。独立逐关节FK与批量FK之间有数值测试，输入图片与manifest通过路径和SHA256核对。

这些检查能验证数据转换和实现一致，不能自动证明URDF轴线、真实关节零点及机械结构完全准确。

### 4.7 问题：关节零偏与相机外参不能分别确定

这是当前两轴结构的关键限制。设：

```text
G(q) = O_yaw R_yaw(q_yaw) O_pitch R_pitch(q_pitch)
G(q + delta) X C_i = Y
```

存在严格等价关系：

```text
B = O_yaw R_yaw(delta_yaw) inverse(O_yaw)
X_effective = R_pitch(delta_pitch) X
Y_effective = inverse(B) Y
G(q) X_effective C_i = Y_effective
```

因此yaw零偏可被未知棋盘基座位姿吸收，pitch零偏可被相机外参吸收。即使观测足以确定等效手眼关系，也不代表能额外确定这两个物理零偏。

本次14个参数为相机6、棋盘6、零偏2，无先验图像观测雅可比秩为12。增加同类型的图像不能消除这两个自由度。

解决方案分两层：

1. 当前数据范围内：给零偏加入以0为中心、尺度1°的约束，只用于选定等价解的坐标约定；导出时把零偏吸收到等效X/Y，继续使用原始`q_ours`。
2. 若需要真实物理零偏：补充足够的外部零位或几何参考，分开首/末关节偏差、相机安装和棋盘位置，再联合标定。

本次零偏加入前后误差几乎相同，正是这个等价关系的体现。不能把接近0的优化偏置解释为“物理零偏已经测准”，也不能宣称RMS下降主要来自零偏校正。

工程实现中还有一个细节：三维残差向量会随yaw规范旋转，虽然它的平方和不变，直接查看该向量的雅可比可能误判秩。因此可辨识性检查使用固定图像坐标的重投影残差，并用数值测试验证变换前后投影及三维代价不变。

### 4.8 问题：训练误差、验证误差和绝对误差混在一起

早期只有少量留出图像，且算法表主要比较同批训练拟合。9月17日和18日固定前80个姿态训练、后20个验证；角点检测前将分组写入`split.json`，验证原点参考中心只由训练组计算。

验证RMS计算为：

```text
P_i = xyz(G_i X C_i [0,0,0,1]^T)
P_reference = mean(P_i for training poses)
error_i = norm(P_i - P_reference)
validation_RMS = sqrt(mean(error_i² for validation poses))
```

全板角点另用训练求得的Y作参考。验证时没有重新求X、K、D，也没有用验证点重新居中。

9月17日追加诊断是在看过留出结果后开展，所以那批后续结果属于开发诊断。9月18日沿用已确定的方法，在观察新验证结果前选定主目标，保留各阶段数值，不根据验证成绩重新选择最终模型。

这仍是同批采集中的姿态留出。要验证跨天、跨距离或不同工作目标，应另做相应测试；外部绝对坐标精度需要足够准确的参考测量。

### 4.9 问题：依赖环境不同，结果难以复现

早期C++角点检测与Python求解使用的OpenCV版本并不完全相同，同一原始数据的基线数值也会有差异。用户环境的NumPy与系统SciPy还曾出现版本支持范围警告。

后续固定系统Python环境，保存OpenCV/NumPy/SciPy版本、输入哈希、算法参数和源代码快照。9月18日导出的XML被重新读取，并独立重算验证RMS，得到相同结果。

因此比较收益时应优先看**同一条实验管线内部**的结果，不把不同版本、不同检测方法、不同分组的差值全部解释为某个算法的提升。

## 5. 本项目一步步改进的实际过程

### 阶段一：9月16日，先排查数据与实现

原640×480数据的棋盘较小；原C++日志的留出验证RMS约30.36 mm。补充检查了图像/位姿匹配、单位、FK、坐标方向以及运动几何。

随后离线试验统一角点处理、进行联合重投影优化，留出原点RMS约13.44 mm，仍未通过。该试验只有5张有效留出图，且与原C++检测/求解版本存在差异，不能把30.36→13.44全部归为单个算法收益。

该阶段的价值是确认：需要同时改进视觉观测和求解目标，不能仅以多种算法结果不一致就断言两轴姿态不可用。

### 阶段二：9月17日，提高原始分辨率并分解优化收益

RGB改为1280×720，100张全部检测成功，训练/验证固定为80/20。但约1 m距离下单格仍只有约13～16 px。

| 9月17日同批试验 | 验证原点 RMS / mm | 验证原点 Max / mm |
|---|---:|---:|
| PARK | 25.423 | 41.480 |
| 联合重投影＋零偏规范约束 | 7.484 | 17.551 |
| 角点精修＋联合重投影 | 6.857 | 15.891 |
| 角点精修＋全板三维优化 | 6.446 | 8.922 |
| 仅板原点三维优化，诊断用途 | 6.375 | 12.024 |

仅原点优化的RMS略低，但最大误差及整块棋盘约束不同，因此没有为了一个更小的数值忽略其他指标。该批所有上述结果都没有达到5 mm。

### 阶段三：9月18日，提高实际棋盘成像尺寸后按固定流程复测

仍使用1280×720和15 mm棋盘，距离变为约0.49～0.57 m，每格约24～28 px。预先选定小窗口精修和全板目标后，留出原点RMS达到3.4242 mm。

从证据上可得出：图像中棋盘的实际尺寸改善，与更好的初值和优化流程共同带来了本次通过结果。不能得出“只要换成720p就必然≤5 mm”，也不能声称在原1 m距离已经完成相同精度验收。

本次没有进行的操作包括：改变15 mm尺度来压低残差、修改验证分组、删除较差验证点、添加逐帧自由关节修正、将验证图加入拟合后沿用旧分数。

## 6. 代码优化落实在哪里

| 文件/功能 | 实现的作用 |
|---|---|
| `src/camera/RealSenseCamera.py` | 默认采集尺寸改为1280×720；沿用实际视频流K读取 |
| `src/camera/run_realsense_viewer.py` | 默认启动参数改为1280×720；未加入新的外部采集功能 |
| `calibrate_head_joint.py::HeadKinematics` | 按URDF计算两轴FK、进行零偏规范换算 |
| `calibrate_head_joint.py::detect` | 检查原图尺寸、哈希、SB检测与可选精修，保存质量统计与缓存 |
| `calibrate_head_joint.py::project` | 在固定K/D下计算整条运动链对应的像素预测 |
| `calibrate_head_joint.py::evaluate` | 明确区分训练/验证、原点/全角点/像素残差，输出逐帧数值 |
| `calibrate_head_joint.py::main` | 预先分组、PARK初始化、分阶段优化、零偏可辨识性检查 |
| `export_head_calibration.py` | 导出统一单位和方向的矩阵、报告、XML、图表、源代码快照 |
| `test_calibrate_head_joint.py` | 验证变换方向、单位、规范不变性和验证数据隔离 |

当前求解器专用于`waist_yaw_Link → head_yaw_joint → head_pitch_joint → head_pitch_Link`的两轴链，并假设一个固定板session。不同机器人、多session移动过的棋盘、其他畸变模型或不同分组策略，不能不改模型就直接拼接输入。

## 7. 后续图像与采集如何优化

建议先改善决定观测质量的条件，再讨论增加模型自由度：

1. **确认真实RGB内参和畸变。** 读取实际启动profile，核实图像是否裁剪、缩放或去畸变。固定可靠参数，不从当前小视角手眼数据自由拟合全部内参。
2. **检查每格实际像素数。** 本次有效改善发生在约24～28 px；20～30 px是采集设计参考，不是通过精度的硬保证。
3. **控制清晰度与曝光。** 停稳再拍；避免反光、过曝、拖影与板弯曲；保留无损原图和独立角点可视化。
4. **多帧测重复性。** 每姿态连续10～20帧，先区分随机抖动和稳定偏差。采用融合时，应在最终应用和验证中采用相同策略；单帧任务不能只报告融合成绩。
5. **保存可靠时间配对。** 记录曝光与关节采样时间及时间域，检查停止后的状态窗口。当前每姿态单图无法独立拆解视觉噪声、同步和机械重复性。
6. **覆盖真实工作范围。** yaw和pitch独立变化，棋盘完整可见。手眼采集期间板相对基座不动；内参验证时才可以移动板。
7. **回访姿态检查机械问题。** 离开后重新到达同一姿态，并比较不同到达方向；检查回差、结构变形、安装松动和漂移。
8. **真实零偏需要外部参考。** 两轴观测缺少的独立约束，不能靠增加重复图片或任意逐帧补偿获得。

完整现场清单见[采集与验收方案](HAND_EYE_CAPTURE_PLAN.md)。图像或模型改善后，应冻结新流程，再采新的最终验证数据，避免反复用同一验证集选择参数。

## 8. 最终矩阵怎么使用

### 8.1 矩阵方向与数值

当前输出为RGB光学系到头部末端：

```text
p_head_pitch_Link = R * p_RGB_optical + t
平移单位：mm

 0.0097186392  0.0318988758  0.9994438502  48.2865847523
-0.9999527729  0.0003161124  0.0097134987  71.5464494058
-0.0000060869 -0.9994910514  0.0319004415 -27.7481986793
 0.0000000000  0.0000000000  0.0000000000  1.0000000000
```

实际使用优先读取文件，不手工抄写截断后的矩阵。变换到基座需额外乘当前姿态的G：

```text
p_base = G(q_ours) X p_RGB_optical
```

若输入点来自深度相机原生坐标系，应先按正确RGB/深度外参换到RGB optical。长度单位统一后再计算，不能将m点直接加mm平移。

### 8.2 读取XML的示例

下面只演示坐标变换，不执行硬件运动。示例点不是实际标定目标，也不是额外精度验证：

```bash
cd /home/robot/hand_eyes_calibration
/usr/bin/python3 -s - <<'PY'
import cv2
import numpy as np

path = "calibration_output_20260918_joint/calib_extrinsic.xml"
fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    raise RuntimeError(f"Cannot open {path}")
R = fs.getNode("R").mat()
t = fs.getNode("t").mat().reshape(3)
fs.release()

p_rgb_optical_mm = np.array([0.0, 0.0, 550.0])
p_head_mm = R @ p_rgb_optical_mm + t
print("示例点在head_pitch_Link中的坐标(mm):", p_head_mm)
PY
```

URDF参考文件使用m与rad，子坐标系明确为RGB optical。实体`camera_link`与光学系轴向不同，可能还存在原点偏移，不能把光学外参直接替换实体安装关节；必须使用完整固定变换换算。

导出结果中的`joint_offsets_to_apply_rad`为零，含义是外参已经换算回原始关节约定。不要再把优化过程中接近零或其他任意规范下的偏置加到电机读数上。

### 8.3 与现有检测程序对接

`detect.cpp`包含从当前目录`calib_extrinsic.xml`读取R、t的逻辑，未找到时还可能使用fallback。实际对接必须确认真正加载的是目标文件、对应RGB内参和当前G，而不能只看检测窗口正常就认为新外参生效。

本文提供的文件和URDF片段尚未部署到运行中的机器人。原有检测程序的整条点云/基座变换链，也不在本次离线固定板验收范围内。

## 9. 数值检查、故障排查与验收

### 9.1 运行数值测试

```bash
cd /home/robot/hand_eyes_calibration
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -s tools/test_calibrate_head_joint.py
```

当前5项检查已经通过：

1. 批量FK与独立逐关节URDF变换组合一致。
2. 两个端点零偏换算到X/Y后，相机观测严格等价。
3. PARK对无噪声合成观测可以恢复已知手眼真值。
4. 含噪情况下，全板三维平方代价在零偏规范变换前后不变。
5. 即使将验证观测人为移远，训练参考中心和训练残差也不会随之变化。

这些检查验证算法实现的重要性质，不替代实机精度验证。

本次还检查了输入文件哈希不变、80/20分组无交叉、优化收敛、旋转正交及行列式、XML单位、URDF换算、导出XML复算的RMS与JSON一致。

### 9.2 常见问题及处理

| 现象 | 检查与处理 |
|---|---|
| 找不到manifest | `--capture`应指向采集根目录，不能指向images |
| 图片尺寸不匹配 | 检查实际图片和camera_info；不要直接改元数据掩盖尺寸变化 |
| 检测数不足 | 确认11×8内角点配置、图像清晰度、完整性和像素尺寸 |
| 重复图像内容报错 | 排查采集是否保存了同一帧，不通过复制文件凑姿态数 |
| q_raw/q_ours不匹配 | 核对单位和方向转换，避免pitch重复取反 |
| 输出平移异常大 | 先核对方向、单位、格距、板是否固定和PnP质量 |
| 数值运行成功但RMS超标 | 阅读goal.passed和逐帧误差；不能把退出码0或SUCCESS当达标 |
| 加入零偏后数值变化、误差不变 | 检查不可辨识性和坐标规范，不直接写入电机零位 |
| 图像重投影小但手眼误差大 | 检查整条运动链、同步、内参与几何条件，单图PnP好不代表闭环好 |
| 部分姿态误差随到达方向变化 | 检查回差、结构和重复定位，补采回访数据 |
| 近距离通过、远距离未通过 | 增加真实远距离观测和独立验证，不外推近距离指标 |
| 换Python后数值或导入错误 | 检查实际解释器和依赖版本，使用记录的兼容环境 |

### 9.3 后续验收要求

如果目标仍是当前约半米范围的固定板RMS≤5mm，本批已有留出结果支持。如果目标升级，应分别补充：

- **最大误差≤5mm：**目前原点最大5.2377mm、全角点最大5.4498mm，尚未通过。
- **1～1.2m范围：**该范围的新验证数据；不能用当前0.49～0.57m结果替代。
- **绝对定位：**坐标已知、测量不确定度足够小的外部参考，并计入完整机器人链误差。
- **深度点云：**深度偏差、尺度、RGB/深度配准与最终三维坐标误差。
- **实际抓取：**在上述基础上增加物体定位、机械臂基座关系和TCP误差验收。

验证集不能因为某些点误差大就删除；修正方法后应保留原失败记录并另采确认集。也不能用全部100姿态重拟合后，继续把原80/20结果当作新矩阵的独立验证成绩。

## 10. 相关文件

- [2026-09-16排查与试验记录](calibration_precision_20260916/README.md)
- [2026-09-17完整报告](calibration_output_20260917_joint/REPORT.md)
- [2026-09-18完整报告](calibration_output_20260918_joint/REPORT.md)
- [最终结构化结果](calibration_output_20260918_joint/calibration_result.json)
- [最终外参XML](calibration_output_20260918_joint/calib_extrinsic.xml)
- [最终逐帧误差](calibration_output_20260918_joint/joint_per_frame.csv)
- [最新误差对比图](calibration_output_20260918_joint/error_comparison.png)
- [现场采集与验收方案](HAND_EYE_CAPTURE_PLAN.md)
- [联合标定程序](tools/calibrate_head_joint.py)
- [结果导出程序](tools/export_head_calibration.py)
- [数值测试](tools/test_calibrate_head_joint.py)
