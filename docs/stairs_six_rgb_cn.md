# 六段旧基线 SMPL + RGB 验证

ORCS 基线为 5c1e246，保留比例重定时、原始缩放遗漏和 legacy seed 行为。
只共享 SONIC 权重契约守卫，不移植人体时间轴或接触补偿。

## 数据

服务器数据根目录：`/home/cqu023/orcs-data/releases/stairs_5c1e246_six_rgb_v1`。
六段从八段版本逐文件复制并校验，不重新生成 seed；原八段保留。
排除 `stair_p1_02afcc5711d8c546` 和 `stair_p1_4f5c901b5adcb903`。
`dataset_manifest.json` 记录源目录、排除理由和全部文件摘要。

## 环境

本地修改，两个仓库分别管理，不自动提交。
本次服务器通过数据目录下的 `validation_code` 快照运行，不修改服务器主仓库。
加载验证使用 `env.sh`，同时设置 ORCS/VIBE 数据与依赖根目录和 Dex1 URDF。
快照环境的 PYTHONPATH 只在当前终端使用，回到主仓库实验前应 unset PYTHONPATH。
依赖仍复用外层 `.venv` 和本地 assets/mocke/rsl_rl；不运行原版依赖同步。

代码提交并拉取后，可在服务器外层 `.venv` 内对两个仓库运行 `python -m pip install --no-deps -e .`，
分别位于外层 ORCS 和内层 vibe 目录执行。先确认 pip 可用；不要通过该步骤替换依赖版本。

## 任务与验证

任务：`Vibe-PerLoco-GrailStairs-ImgFeat-Ext-Smpl`。
SMPL encoder/SONIC 原始 decoder 与 Theia 冻结；仅视觉提取器、LoRA 和 critic 训练。
相机沿用 ViBe 112×63、垂直视场 42.5°，未经实机标定。
恢复契约检查数据文件、相机、视觉骨干标识、Dex1 URDF/网格及 SONIC 基础权重。

服务器复制好的数据不应原地更改；否则恢复契约将拒绝旧 checkpoint。
运行记录在 `rgb_check_report.json` 和 `ppo_validation/report.json`，以报告实际状态为准。

```bash
cd /home/cqu023/ViBe-SONIC
source /home/cqu023/orcs-data/releases/stairs_5c1e246_six_rgb_v1/env.sh
.venv/bin/python /home/cqu023/orcs-data/releases/stairs_5c1e246_six_rgb_v1/validate_stairs_rgb.py
```

验证入口固定 64 环境、20 轮，然后从明确 checkpoint 恢复 2 轮；输出目录存在时拒绝覆盖。
它检查零 LoRA 一致性、初始化软限位裁剪、冻结参数不变、可训练参数更新及黑图像对动作影响。
保存六张 RGB 截图用于检查相机；短程通过不等于动作质量或实机可靠性通过。
不启动 4096 环境长期训练。

## 本轮实测结果

- 六环境覆盖 clip 0–5，前向/step 通过；tokenizer 840、本体 930、相机 112×63。
- 64 环境 20 次 PPO 更新完成；零 LoRA 与基础 decoder 最大误差 0。
- LoRA、视觉提取器和 critic 更新；SONIC 与 Theia 摘要不变。
- 同一机器人/人体观测下，将 RGB 换为黑图并重新编码，最大动作差 0.2741169。
- 从 `ppo_validation/train/verified_20.pt` 恢复 2 次更新通过。
  runner 保存最后一次迭代编号，加载编号19，恢复执行19和20；实际更新数为20+2。
- 训练 RSI 包含已有±0.05rad关节噪声和软限位裁剪；验证按该规则检查，不改变配置。
- Theia缓存提交：8ce99426c269617346c30112817f0e694e3cde2d。
- 图像样例主要是近处地面，关键上楼阶段台阶可见性尚需专项回放确认。
- 运行仍存在较大跟踪误差和终止；本轮没有验证成功爬楼。

验证中两次RSI断言误设、一次摘要函数不支持标量的失败报告均保留；最终报告为passed。
