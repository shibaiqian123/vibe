# RGB 对照回放

只用于已训练 checkpoint 的交互评估，不改变训练配置、数据、seed 或模型权重。

## 启动

服务器沿用六段数据的 `env.sh`，但将独立验证版 ViBe 放在 `PYTHONPATH` 最前。

```bash
cd /home/cqu023/ViBe-SONIC
source .venv/bin/activate
source /home/cqu023/orcs-data/releases/stairs_5c1e246_six_rgb_v1/env.sh
export PYTHONPATH=/home/cqu023/orcs-rgb-compare-20261019-v1/src:$PYTHONPATH
play Vibe-PerLoco-GrailStairs-ImgFeat-Ext-Smpl \
  --checkpoint-file /home/cqu023/ViBe-SONIC/logs/rsl_rl/vibe_perloco_grail_stairs_smpl/2026-10-18_19-45-38_stairs_six_rgb_release_4096/model_9999.pt \
  --num-envs 6 --viewer viser --rgb-mode normal --rgb-seed 0
```

等待日志中的实际监听端口，沿用对应 SSH 转发。不要停止训练或 TensorBoard；如果端口被旧回放占用，只停止经核实的旧回放进程。更新现有源码后，仍须重启回放；旧进程不会自动加载新功能。

本地源码版本为本次未提交修改；服务器目录是独立代码副本。它与原训练用的 `validation_code/vibe` 分开，未覆盖服务器仓库。正式采用 Git 同步后将路径替换为你实际同步的 ViBe `src`，注意 `env.sh` 会重新设置 `PYTHONPATH`。

## 面板

展开 **RGB 对照**，选择模式，点击 **应用并重新开始**：

- `normal`：原图。
- `gray`：全部像素 127（uint8），然后照常归一化并进入 Theia。
- `shuffle`：环境间固定错配，来自不同片段，固定 seed 决定排列。必须至少有两个片段，且任何片段的环境数不能超过一半；否则报错，不降级。

GUI 只提交请求，主线程应用并重置所有环境。暂停状态保持不变，不会因切换而执行动作。三个模式使用相同重置随机种子；全局仿真步号不会归零，界面同时显示片段局部帧号。

**原始相机 RGB** 与 **策略实际输入 RGB** 是同一次策略前向的快照。Camera Feeds 仍是实时原图。错配模式标明图像来自哪个环境和片段。注意力图叠在策略实际输入快照上；模式重置后等待新策略前向，不显示旧注意力。

对照期间不要热切换 checkpoint。先运行同一片段、同一 checkpoint 的三个模式，重点看卡住、跌倒和跟踪变化，不把热图变化等同于视觉有效性。当前是训练集上的交互诊断，没有批量成功率结论。

## 验证记录（2026-10-19）

服务器：`/home/cqu023/orcs-rgb-compare-20261019-v1/`。

- `check2/report.json`：六环境，每种模式 8 个控制步。六段均覆盖，各片段从首帧开始（报告 start_frames 是拼接数据的全局索引）。
- 正常模式相对于不启用控制对象的动作最大误差：0。
- 三种模式重置后的关节状态最大差异：0。
- actor（含 LoRA）与 Theia state_dict 哈希均未变化。
- `check2/{normal,gray,shuffle}_{raw,input,attention}.png`：实际输入与注意力截图。
- 未改动运行中的训练，不提交或推送。

本机默认 Python 无 PyTorch：只做语法/差异检查；依赖 PyTorch 的单元测试与 GPU 短回放在服务器独立副本执行。已有 `test_stairs_visual_contract.py` 测试缺少当前必填 visual_contract_id，为已有测试夹具问题，单独记录；本次真实 checkpoint 加载及权重不变验证已通过。


最终新增 RGB 单元测试 **15 项通过**。额外旧测试 14 项在未修改的训练代码路径上也同样失败：13 项因 Repose/Cube 任务未注册，1 项因旧测试夹具缺少 visual_contract_id。本轮不修改这些无关任务或夹具，不宣称完整旧测试套件已通过。

实际输入截图（112×63，保存的原始分辨率）：

- 正常：![normal](assets/rgb_compare/normal_input.png)
- 全灰：![gray](assets/rgb_compare/gray_input.png)
- 跨片段错配：![shuffle](assets/rgb_compare/shuffle_input.png)

这些是短回放中的输入证据，不是三个模式的完成率对比。完整机器可读结果见 `assets/rgb_compare/report.json`。
