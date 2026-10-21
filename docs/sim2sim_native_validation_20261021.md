# 三模型原生 MuJoCo 验证记录

## 结论

已实现训练同款 G1＋固定 Dex1、训练相机、原楼梯数据的独立原生 MuJoCo 三模型闭环。不是原 SONIC 43DOF/C++/DDS 入口，TensorRT 尚未接入。
六段初始化输入对照通过；正常 RGB 五段到达末帧，一段下楼在第 169 步失稳停止。不能宣布全部六段动作质量通过。
关注样本正常 RGB 完成 498 个控制步，全灰输入第 364 步触发停止；这是单次确定性对照，不是统计成功率。

## 数值接口与版本

原 checkpoint SHA256：`91dde64705bf9f135708914590e31f4d6ddb6d9dd81959af44f2650851dbc4c9`。
六段初始化及关注样本第 228 帧（共 8 份快照，包括一份重复基线）与原 PyTorch actor 直接对照：全部通过，latent 完全一致，最大 action 误差 1.90734863e-06，门槛保持 1e-5。
Theia 中间 patch 特征 FP32 误差单独记录；最终门槛是完整输入链路产生的 action，未通过扩大 action 容差验收。
每个 case.json 保存 seed、SMPL、motion、来源文件哈希及相机信息。每轮 validation.json 保存具体代码和模型摘要。原始数据、seed、权重未修改。
服务器只使用独立同步代码副本；本地新增文件尚未提交或推送。

## 性能

最初默认 ONNX 线程池首段无窗口吞吐约 0.70×；限制每个 ONNX session 为单线程后，同起点完整回放约 1.68×。
最终 GUI、RGB 窗口、人体骨架均开启，显示限制 15 Hz，策略相机与控制仍 50 Hz；9.960 秒仿真耗时 9.978 秒，实时倍率 0.998211。
控制采用累计单调时钟截止时间；物理每步 5ms，四步一次策略。无需 TensorRT 即已在本次 GUI 测试达到约实时；不代表其他 GPU/桌面负载也一定相同。

## 逐段实测

- `normal_stair_p1_96ceca214c8d1b6a`：rollout_completed_not_quality_certified，498 步；平均点跟踪误差 0.254 m；全局最小接触距离 -13.9 mm；吞吐 1.563×。
- `normal_stair_p1_daa291a807fb834a`：stopped_unstable，169 步；平均点跟踪误差 0.783 m；全局最小接触距离 -23.9 mm；吞吐 1.602×。
- `normal_stair_p2_2583aaa3a5ca8eed`：rollout_completed_not_quality_certified，498 步；平均点跟踪误差 0.112 m；全局最小接触距离 -14.6 mm；吞吐 1.665×。
- `normal_stair_p2_3cf8005eeb361db1`：rollout_completed_not_quality_certified，498 步；平均点跟踪误差 0.111 m；全局最小接触距离 -16.5 mm；吞吐 1.668×。
- `normal_stair_p2_6768415cac75882b`：rollout_completed_not_quality_certified，498 步；平均点跟踪误差 0.124 m；全局最小接触距离 -17.2 mm；吞吐 1.676×。
- `normal_stair_p2_e53716842a1e1f27`：rollout_completed_not_quality_certified，498 步；平均点跟踪误差 0.111 m；全局最小接触距离 -17.1 mm；吞吐 1.709×。
- `gray_stair_p1_96ceca214c8d1b6a`：rollout_completed_not_quality_certified，498 步；平均点跟踪误差 0.196 m；全局最小接触距离 -12.6 mm；吞吐 1.554×。
- `gray_stair_p2_2583aaa3a5ca8eed`：rollout_completed_not_quality_certified，498 步；平均点跟踪误差 0.180 m；全局最小接触距离 -15.6 mm；吞吐 1.754×。
- `gray_stair_p2_3cf8005eeb361db1`：rollout_completed_not_quality_certified，498 步；平均点跟踪误差 0.201 m；全局最小接触距离 -15.5 mm；吞吐 1.728×。
- `gray_stair_p2_6768415cac75882b`：stopped_unstable，364 步；平均点跟踪误差 0.422 m；全局最小接触距离 -17.7 mm；吞吐 1.734×。
- `gray_stair_p2_e53716842a1e1f27`：rollout_completed_not_quality_certified，498 步；平均点跟踪误差 0.174 m；全局最小接触距离 -28.3 mm；吞吐 1.697×。

未对正常模式已失稳的下楼段继续灰图对照。全局接触距离包含所有接触，不等同于脚底穿透指标。
本轮停止启发式包括非有限状态和 pelvis 朝上方向投影小于 0.25；未触发不等同于通过接触、平衡或实机安全验收。

## 已定位的接入问题

- 初版角速度误差来自 MuJoCo BODY 局部惯性轴与训练 pelvis 连杆轴不同，已改为世界角速度旋转到连杆坐标系。
- 两个下楼场景在 Warp 准备阶段需要更多接触与约束槽位；独立准备器设置 nconmax 至少 512、njmax 至少 4096。只改内存容量，未修改碰撞几何、物理步长或求解容差；旧失败日志保留。
- 显示窗口独立限帧，避免阻塞 50 Hz 控制。RGB 通过现有 Tk/Pillow 展示，没有安装 OpenCV 或改动训练依赖。

## 测试及剩余限制

本地 4 个契约单元测试通过：历史回填/顺序、RGB→BGR 不修改源缓存、累计截止时间、角速度连杆坐标变换。三个新增 Python 模块语法检查通过。
仍需定位下楼段 stair_p1_daa291a807fb834a 的独立动力学失稳；其余五段也有约厘米级全局接触负距离，不能只凭完成回放认定动作质量合格。
源数据保留 5c1e246 旧基线缩放/重定时限制。未验证独立部署对所有后续帧的逐帧 PyTorch 等价性；当前数值对照是初始化与第 228 帧，同一图片输入。
尚未接 TensorRT、VR/VLA、实机相机或 SONIC 43DOF；原 SONIC 真机部署不变。

## 文件入口

服务器目录：`/home/cqu023/sonic-diagnostics/vibe-native-v1`。其中 `code/README_zh.md` 为启动说明，`summary.json` 为结构化结果，`checkpoint_comparison.json` 为直接 checkpoint 对照。
关注样本场景：`case_stair_p2_6768415cac75882b`；最终 GUI 运行：`gui_focus_02`。运行日志和 trajectory.jsonl 均保留。
