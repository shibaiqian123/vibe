# 三模型原生 MuJoCo sim2sim

此入口运行 `motion_encoder.onnx`、`visual_decoder.onnx` 和 Theia ONNX。它不使用
SONIC 原版 `run_sim_realtime.py`，也不需要启动 C++ 部署或 DDS。采用训练同款 G1
固定 Dex1、训练相机和原数据；不使用 SONIC 43DOF 替换模型。

## 环境与入口

代码在独立 ViBe 仓库 `src/vibe/sim2sim/`；无需修改现有训练入口。
先激活外层 ORCS `.venv`，设置六段数据环境。需要同版本 ORCS、mocke、mjlab、
MuJoCo、ONNX Runtime、Pillow，以及 GUI 回放时的 Tk/Pillow。不要用部署 extras 安装
替换正在使用的 rsl_rl fork。当前服务器 ORT 位于独立验证目录，仍属临时依赖，
以下命令明确保留此路径。

```bash
cd /home/cqu023/ViBe-SONIC
source .venv/bin/activate
source /home/cqu023/orcs-data/releases/stairs_5c1e246_six_rgb_v1/env.sh
export PYTHONPATH="/home/cqu023/ViBe-SONIC/vibe/src:/home/cqu023/ViBe-SONIC/src:/tmp/orcs-split-validation-20261020/runtime"
export BUNDLE=/home/cqu023/orcs-exports/stairs_split_v1/focus/split_sonic
export VISION=/home/cqu023/orcs-exports/stairs_split_v1/vision/theia-tiny.onnx
```

服务器首轮实现使用 `/home/cqu023/sonic-diagnostics/vibe-native-v1/code/` 的明确同步副本。
仓库未提交、未推送，不能假定服务器仓库已有这些新增模块。

## 准备一个独立场景

输出目录必须不存在；不会覆盖已有实验。初始化可以指定 `--frame 228`。

```bash
python /home/cqu023/sonic-diagnostics/vibe-native-v1/code/prepare.py \
  --bundle "$BUNDLE" \
  --data-root "$ORCS_DATA_ROOT" \
  --family stair_p2_6768415cac75882b \
  --frame 0 \
  --out /home/cqu023/sonic-diagnostics/my_native_case
```

准备器只利用训练任务解析资产、读取 seed、执行 reset 和生成观测对照；不运行 PPO。
`scene.mjb` 内嵌碰撞、网格和原生 PD 执行器；`initial.npz` 保存 reset 后实际状态、
动作历史初值、训练观测和相机图像；`reference.npz` 保存同一人体参考与任务查询。
`case.json` 记录原始数据校验值、资产和相机、初始化帧、代码来源。
随机扰动在准备对照时关闭。历史按训练语义用 reset 帧回填十帧，不伪造过去十帧运动。

## 检查与回放

```bash
python /home/cqu023/sonic-diagnostics/vibe-native-v1/code/runtime.py \
  --case /home/cqu023/sonic-diagnostics/my_native_case \
  --bundle "$BUNDLE" --vision-encoder "$VISION" \
  --provider cpu --ort-threads 1 --check-only \
  --out /home/cqu023/sonic-diagnostics/my_native_check
```

每次正式回放也先执行同状态检查。使用同一 RGB，分别比较独立状态观测与训练
快照观测，并比较训练视觉特征和 Theia ONNX 特征输入 decoder 的最终动作。
动作门槛保持 `1e-5`，量化 latent 也单独检查；视觉中间特征误差逐项报告，
不把特征绝对误差直接当作动作误差。该检查复用已验证的导出模型，不能替代
所有帧的 checkpoint/PyTorch 对照。

```bash
# GUI 在服务器图形桌面；不是 Viser 网页。需要可访问的 DISPLAY。
export MUJOCO_GL=glfw
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
python /home/cqu023/sonic-diagnostics/vibe-native-v1/code/runtime.py \
  --case /home/cqu023/sonic-diagnostics/my_native_case \
  --bundle "$BUNDLE" --vision-encoder "$VISION" \
  --provider cpu --ort-threads 1 --steps 10000 \
  --viewer --realtime --hold-final \
  --out /home/cqu023/sonic-diagnostics/my_native_play
```

空格暂停/恢复；运行中 R 重置到本 case 的初始化帧；关闭 MuJoCo 窗口结束。
参考结束后停在末帧，`--hold-final` 保留画面供查看。人体只画在 viewer 的诊断层，
不会进入碰撞或策略相机图像。实际 Theia RGB 另窗显示。

无窗口运行使用 `MUJOCO_GL=egl`，不传 `--viewer --hold-final`。
加 `--rgb-mode gray` 可做相同初始化的全灰对照。每次必须使用新的 `--out`。

## 控制与性能

- 原生 MuJoCo 物理步长 5 ms，四个物理步后更新一次策略（50 Hz 仿真时间）。
- SMPL 经原 mocke tokenizer 进入 motion encoder；Theia ONNX 输入为 BGR uint8，
  MuJoCo RGB 在入口转换一次，图像归一化在 ONNX 内执行。
- 本体角速度由世界角速度转换至 pelvis **连杆**坐标系，不直接使用 BODY 局部惯性系速度。
- 930 维历史按观测项分组、每项从旧到新排列；关节目标、PD 和力矩限制均来自训练模型。
- `--realtime` 用累计单调时钟截止时间等待；运行慢时不跳物理步、不按墙钟加速参考。
- NVIDIA 渲染环境变量不选择 ONNX GPU。`--provider cuda` 要求实际安装 CUDA provider，
  缺失时报错，不静默降级。当前已测 CPU；`--ort-threads 1` 限制小批量推理线程竞争。

## 报告边界

`validation.json` 记录同状态误差、使用的 provider、实际仿真时间/墙钟时间和模型摘要。
`trajectory.jsonl` 记录跟踪、姿态、最小接触距离、力矩饱和比例、动作与状态。
`rgb_*.png` 是实际送入 Theia 的图像。参考结束与数值稳定不等于动作质量合格：
报告状态 `rollout_completed_not_quality_certified` 特意保留此区别。
全局最小接触距离包含所有接触，不等同于脚底穿透；需要结合回放判断。

保持源数据、seed、模型及真机代码不变。此版本仍保留六段旧基线的数据限制，
尚未验证真实相机、VR/VLA、SONIC 43DOF 或实机迁移。

## 复用来源

- 本项目 `vibe.export.agent.split_sonic.SplitOnnxAgent`：模型哈希、manifest、ONNX 装配。
- `mocke.sonic.mdp.observations.sonic_smpl_tokenizer`：原始 SMPL 特征与未来窗口。
- ORCS `robot_root_*_vel_cmd` 与 seed-backed RSI：准备阶段直接读取原实现的结果。
- SONIC `gear_sonic/scripts/run_sim_loop.py` 的原生 MuJoCo 回放结构，以及独立
  `sonic-realtime-v1` 诊断中验证的累计截止时间方式。未复制真机通信代码。
- MuJoCo 官方 `Renderer`、`viewer.launch_passive`、`mj_step`；沿用当前仓库许可证。

## 本次验证与 TensorRT 边界

完整验证记录见 `sim2sim_native_validation_20261021.md`。原 checkpoint 与独立三模型链路在八份同状态快照上对照，量化 latent 完全一致，最大 action 误差为 1.91e-6。正常 RGB 五段到达末帧，一段下楼第 169 步失稳；不代表六段动作质量全部通过。

显示窗口默认限制为 15 Hz（`--viewer-hz 15`），策略相机和控制保持 50 Hz。关注样本开启两个窗口的实测为 0.998× 实时；这不是其他机器或负载下的性能保证。

现有 SONIC C++ / TensorRT 与 `.venv_teleop` 保持不变。此入口使用外层 ORCS `.venv` 和 ONNX Runtime，当前没有 TensorRT 后端。后续若使用 TensorRT，需要为三模型建立对应引擎、接入视觉特征和查询，并重新验证 FSQ latent 与 action 的数值一致性；不能直接用旧 SONIC decoder 接口替换视觉 decoder。
