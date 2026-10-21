# ViBe 独立部署工程（接入中，不能替代真机部署）

本目录提取 SONIC 三终端所需代码，并已完成 C++ TensorRT 三模型适配与无动作数值验证。
**当前不是已完成的三终端控制程序。没有声称键盘行走、VR 或 DDS 闭环已通过。**
原 SONIC 源码、真机二进制、low-latency 模型、训练进程均未修改；不自动提交推送。

## 当前可用内容

- `vendor/sonic/`：逐文件保留原样的 SONIC C++ 部署、键盘、manager、planner、PICO 与仿真桥接源代码，带原许可证；并非可直接用视觉模型启动的成品。
- `vendor/sonic_diagnostics/`：之前仅仿真的时钟修正源码，供适配时复用。它仍有原目录依赖，不要当作新场景启动器。
- `controller/`：直接复用 SONIC TRTInferenceEngine 的多输入绑定、静态形状/类型校验、三模型数值测试。
- `sim/rgb_transport.py`：RGB 传输及重置编号、步号、时间戳匹配检查；尚未挂接 DDS 循环。
- `input/`：复用原 SONIC 消息打包器的有界解码器及只读流检查；不发送机器人命令。
- `configs/sim_only.json`：独立回环端口、DDS 域及尚未就绪的输入路径。配置不是控制程序实现。

## 环境与编译

使用已有 ORCS `.venv` 做模型数值验证，不覆盖 `.venv_teleop`。只读检查：

```bash
python deploy/scripts/preflight.py
```

当前服务器 ORCS 环境缺少 `unitree_sdk2py` 和 `zmq`；预检查返回非零是准确的未就绪状态。不要通过改 JSON 的 status 来绕过它。

复用已安装的 CUDA/TensorRT 编译独立 C++ 库：

```bash
export CUDAToolkit_ROOT=/usr/local/cuda-12.1
export VIBE_DEPLOY_BUILD=/home/cqu023/sonic-diagnostics/vibe-deploy-build-v1
bash deploy/scripts/build_trt.sh
```

构建器保留 vendored 源文件不变，在构建目录生成副本，仅关闭 TF32，并将 TensorRT 版本及适配版本加入原引擎缓存标识；不使用 `-ffast-math`。

## 三模型验证命令（不发送动作）

以下从 ViBe 根目录执行；`--out` 必须是新目录。

```bash
python deploy/controller/validate_trt.py \
  --bundle /home/cqu023/orcs-exports/stairs_split_v1/focus/split_sonic \
  --vision /home/cqu023/orcs-exports/stairs_split_v1/vision/theia-tiny.onnx \
  --case /home/cqu023/sonic-diagnostics/vibe-native-v1/case_stair_p2_6768415cac75882b \
  --library /home/cqu023/sonic-diagnostics/vibe-deploy-build-v1/libvibe_trt.so \
  --out /home/cqu023/sonic-diagnostics/my_deploy_validation
```

临时 ONNX Runtime 依赖在 `/tmp/orcs-split-validation-20261020/runtime`，需要按之前的验证环境加入 PYTHONPATH。原模型只读；TensorRT 缓存、Theia 输入适配副本均写到本次输出目录。

Theia 原输入是 BGR uint8。服务器 TensorRT 解析器拒绝 UINT8，因此创建独立的 BGR float32 像素输入副本；值仍为 0～255，只在原图中完成一次归一化。只允许已检查的 Slice/Transpose/Cast 输入边界，随机图和灰图 ORT 输出必须完全一致。

## 三终端的提取位置与剩余接入

1. 仿真：`vendor/sonic/gear_sonic/utils/mujoco_sim/` 保留状态发布与命令接收。待接训练场景、RGB side channel 和同一仿真时钟。不能直接把 29 个训练关节按数组位置填入另一套顺序。
2. 部署：`vendor/sonic/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/` 保留原状态机和所有输入处理器。待将三模型绑定接入观测/控制循环，替换旧低延迟动作配置，并完成站立接管、暂停、重置及断流验证。
3. 输入：保留原直接键盘与 `zmq_manager` 两条路径，必须互斥。PICO 源码在提取目录中，录制 SMPL 发布器尚未接入。没有假装“键盘发出了 command”就等于键盘走路可用。

协议实际头长为 **1280 字节**，原文件注释中的 1024 已过时；代码读取原打包器常量，不依赖该注释。

只读检查已运行 manager 的消息（默认独立 15556 端口）：

```bash
python deploy/input/inspect_stream.py --port 15556 --seconds 10 --out /tmp/sonic_input_audit.json
```

`command` 是模式/启动停止，`planner` 是运动意图，`pose` 是运动帧。PICO 发送 SMPL、根姿态、腕部关节和 VR3PT，但仍须核对未来窗口和视觉查询。840 维 SMPL encoder 不接受原 planner 机器人参考或 VR3PT 数据。

当前找到的完整 SONIC encoder 属于 low_latency 与 sonic_v1_1；尚无证据证明其全部模式与当前 release 视觉 decoder 的 latent 契约配套。不要直接替换文件名启用。

## 验证状态

详见 `VALIDATION_zh.md`。当前验证范围是一个初始化状态、正常/灰图、三模型 TensorRT 对 ORT；不是三终端闭环或 VR 成功。

测试：`python -m unittest discover -s deploy/tests -v`。
