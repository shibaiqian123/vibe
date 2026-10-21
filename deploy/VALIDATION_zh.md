# 本轮验证与未完成项

## 已完成

- SONIC 部署与相关输入/仿真源码提取，原样文件校验记录见 SOURCES.json。
- SONIC 原 TRTInferenceEngine 的 C++ 多输入适配库成功编译。
- Theia 原 uint8 输入在服务器解析器失败的日志保留；独立 float32 像素副本与原模型在随机/灰色图像上特征最大误差均为 0。
- 关注样本初始化状态：正常 RGB 的 action 最大误差 6.556510925292969e-7；灰图为 7.152557373046875e-7。两种模式 latent 完全一致，门槛保持 1e-5。仅代表这两个同状态测试，不外推整段。
- 消息格式及 RGB 回合/时间匹配的 12 项单元测试通过。

## 服务器证据

构建目录：`/home/cqu023/sonic-diagnostics/vibe-deploy-build-v1`。
原失败日志：`/home/cqu023/sonic-diagnostics/vibe-deploy-trt-v1.log`。
成功报告：`/home/cqu023/sonic-diagnostics/vibe-deploy-validation-v3/validation.json`。
成功日志：`/home/cqu023/sonic-diagnostics/vibe-deploy-trt-v3.log`。
本轮均为无动作验证，没有启动新的 DDS 控制闭环。

## 尚未完成，不能称为三终端部署成功

- 训练同款场景与 SONIC DDS 桥接、RGB 通道的运行集成。
- C++ 状态机内 930 维历史、任务查询、三模型推理与动作输出的完整连接。
- 默认站姿到策略接管、暂停/重置、断流处理的实际验证。
- 录制 SMPL 三终端短程及整段回放。
- keyboard planner、VR3PT 与完整 SMPL 的独立兼容验证，以及真实 PICO 数据测试。

当前 ORCS 环境实测缺少 unitree_sdk2py、zmq，不能直接拿 teleop 环境的包覆盖训练环境。需要独立部署依赖方案。
现有完整 encoder 只发现 low_latency、sonic_v1_1 两套；不能默认它们与当前 release 视觉 decoder 配套。

因此未给出可接管的“新三终端启动 SOP”，也没有把原 SONIC 启动脚本改名冒充新的视觉部署入口。
