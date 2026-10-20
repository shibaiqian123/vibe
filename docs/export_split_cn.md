# 三模型拆分导出

本功能复用 ViBe 官方 `export-agent` 的严格 actor 加载、manifest、DualPolicy 与两环境检查。
动作编码与量化使用 rsl_rl 原 `_OnnxSonicBaseModel` / `FoldedFSQ`；视觉部分继续使用
`_OnnxExtractorSonicAdapterModel` 和 `merge_adapters`，不重新实现网络。
SONIC 的 encoder/decoder 划分作为接口参考，SONIC 原生 checkpoint 导出器不能直接加载 ViBe checkpoint。

参考来源：
- https://github.com/lok-i/vibe/blob/main/docs/export.md
- https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/user_guide/training.md#onnx-export
- 当前依赖中的 `rsl_rl/models/sonic_base_model.py`、`extractor_model.py`、`_onnx_export.py`

只改独立 ViBe 仓库，不改 ORCS、rsl_rl 或训练权重。上游私有 wrapper 类型不匹配时明确失败。

## 环境与导出

在服务器的已安装 ViBe/ORCS 环境执行。首先确认 `vibe.__file__` 指向本次同步版本；
已有 `env.sh` 可能将 PYTHONPATH 指向旧 validation_code，必须显式将新 ViBe src 放在前面。
不要运行会替换 ORCS/rsl_rl 的依赖同步流程。需要已有 torch、onnx、onnxruntime、mjlab 及项目依赖。

```bash
cd /home/cqu023/ViBe-SONIC
source .venv/bin/activate
source /home/cqu023/orcs-data/releases/stairs_5c1e246_six_rgb_v1/env.sh
export PYTHONPATH=/home/cqu023/ViBe-SONIC/vibe/src:$PYTHONPATH
python -c 'import vibe; print(vibe.__file__)'

export-agent Vibe-PerLoco-GrailStairs-ImgFeat-Ext-Smpl \
  --split-sonic \
  --export-family stair_p2_6768415cac75882b \
  --checkpoint-file /home/cqu023/ViBe-SONIC/logs/rsl_rl/vibe_perloco_grail_stairs_smpl/2026-10-18_19-45-38_stairs_six_rgb_release_4096/model_9999.pt \
  --output-dir /home/cqu023/orcs-exports/stairs_split_v1/focus \
  --provider cpu --check-steps 128 --closed-loop-min-steps 128

export-encoder --tag theia-tiny \
  --out /home/cqu023/orcs-exports/stairs_split_v1/vision
```

输出目录 `focus/split_sonic/` 包含 `motion_encoder.onnx`、`visual_decoder.onnx`、各自 manifest、
`bundle.json` 和 `validation.json`。已有 split_sonic 目录时拒绝覆盖。失败诊断留在
`.split-pending-*`，状态 failed；`--no-check` 为 unvalidated，不能当作验收通过。
Theia 独立导出沿用官方 FP32、112×63 配置及其自己的误差门槛。

`--export-family` 同时收缩地形网格与 command 的 tile keys，不改任何源文件。
逐段检查六段时，每段使用不同的输出目录，并依次替换 family：

- stair_p1_96ceca214c8d1b6a
- stair_p1_daa291a807fb834a
- stair_p2_2583aaa3a5ca8eed
- stair_p2_3cf8005eeb361db1
- stair_p2_6768415cac75882b
- stair_p2_e53716842a1e1f27

不指定 family 时保留原网格，两环境运行不代表覆盖六段。完整片段回放应提高 check-steps，
并按报告区分片段结束重置与跟踪失败；不通过降低门槛宣称通过。

## 接口与离线串联

- motion encoder：原 tokenizer 观测 → 量化后 float32 `[1,64]`。没有额外量化或重排。
- visual decoder：latent + 本体历史 + 图像 tokens/CLS + manifest 列出的任务/本体查询与辅助条件。
  不接受仅一个 latent；额外输入不可省略。输出 action 尚需按 manifest 转成关节目标。
- Theia：带 batch 的 BGR HWC uint8 `[1,63,112,3]`；预处理已在图中。相机 RGB 需在入口换通道一次。

用 ONNX 端口名保存 float32 NPZ 输入，提供原 tokenizer 或直接提供量化后 latent：

```bash
python -m vibe.export.agent.run_split \
  --bundle /home/cqu023/orcs-exports/stairs_split_v1/focus/split_sonic \
  --inputs /path/to/named_observations.npz \
  --vision-encoder /home/cqu023/orcs-exports/stairs_split_v1/vision/theia-tiny.onnx \
  --image-bgr /path/to/batched_bgr_uint8.npy
```

可不提供图像参数，直接在 NPZ 中提供已计算的视觉特征。模型文件哈希在加载时校验；
Theia 的模型名、输入颜色和形状检查不能代替权重版本验证，部署前须固定训练时相同 HF revision。
manifest 中的查询和历史仍由调用端产生；这里没有隐含的 SMPL→辅助查询转换，也不是实机程序。

## 验证边界

测试使用真实上游 ExtractorSonicAdapterModel、非零适配器及官方 ONNX wrapper。
特征置零测试只检查视觉计算没有丢失，不等同于全灰 RGB 经 Theia 的端到端评估。
真实 checkpoint 检查逐步比较 FSQ latent 和动作，默认动作误差门槛 1e-5。
两环境仍使用 mjlab；不是独立 MuJoCo、VLA 或实机部署验收。
