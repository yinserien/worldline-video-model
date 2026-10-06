# 性能选项与基准测试

包内所有加速项都是**可选**的，默认走参考实现（FP32、reference 注意力、普通 AdamW、
不编译、不固定内存）。打开任何一项都不改变默认数值契约；下面写的是实现边界，不是性能
承诺：具体收益取决于设备、torch 版本、batch 与数据形状，请自己测。

## performance 段

配置里的 `performance`（0.3.0 新增，默认为参考值）：

| 字段 | 默认值 | 取值 | 作用与边界 |
|---|---|---|---|
| `precision` | `float32` | `float32` / `bfloat16` | 模型**计算**精度。bfloat16 通过 `torch.autocast` 生效；持久状态仍为 FP32，时钟仍为 float64，NLL/KL、投影与标准化、像素指标一律在 FP32 计算（见下） |
| `anchor_attention` | `reference` | `reference` / `sdpa` | `reference` 走显式注意力并返回权重；`sdpa` 用 `scaled_dot_product_attention` 融合核（带学习到的空间偏置），**不返回权重**。读序列很短，SDPA 不保证更快 |
| `fused_optimizer` | `false` | bool | 仅 CUDA：`torch.optim.AdamW(fused=True)`。CPU 上请求会直接报 `PerformanceError`，不会静默退回 |
| `compile` | `false` | bool | 只编译 **predictor head** 这个纯函数（不是整个循环动力学，也不是解码器）。**是否可用取决于 torch 版本、编译后端与平台**：可用时会先跑一次真实执行（预检）并记为 `applied`+`verified`；后端不可用（例如缺少 C++ 工具链）会抛 `PerformanceError` 并提示关闭该开关 |
| `pin_memory` | `false` | bool | 仅 CUDA：主机侧 pinned 缓冲 + 非阻塞拷贝。数值与不固定时完全一致 |
| `non_blocking` | `false` | bool | 同上，异步 H2D；CPU 上两项都无效果 |

所有值在 `RunConfig.validate()` 阶段检查；未知值直接报错。

### 数值契约（不会因为加速而改变）

- **持久状态**：`slots`/`velocity` 始终 FP32，`time` 始终 float64。bfloat16 只影响中间
  激活（`promote_state` 会把状态提升回 FP32）。
- **概率与投影**：`gaussian_nll_bits`/`gaussian_kl_bits`、`LatentProjection.forward`、
  `fit_standardisation` 都在关闭 autocast 的上下文里按提升后的 dtype 计算（FP32；显式
  FP64 输入保持 FP64）。`torch.autocast` 会覆盖显式 dtype，所以只提升张量是不够的。
- **指标**：训练日志、`metrics.json`、解码器 `l1/mse/psnr` 都在 FP32 汇总。
- **验证**：`evaluate_model` 固定用 FP32，作为跨精度可比的标尺。
- **编码器**：冻结的预训练编码器始终 FP32，其精度不在 `performance` 里，也不进缓存键，
  因此既有 cache 与解码器兼容性记录不受影响。
- **注意力**：`model.predict(...)` 的公开默认 `want_attention=True`，即使配置了 `sdpa`
  也会走 reference 读以返回真实权重；训练、验证与推理内部路径显式传
  `want_attention=False` 才使用配置的核。`SpatialAnchors.write` 的语义与实现与注意力
  模式无关。

### 检查点与续训

checkpoint 记录**实际生效**的策略（精度、注意力核、fused 是否真的启用、compile 状态、
pin 是否真的生效）。续训时：

- 语义项（precision / anchor_attention / fused / compile）不一致 → 报 `PerformanceError`；
- 旧 checkpoint 没有该记录 → 视为参考策略，只能以参考设置续训；
- 设备类型、pinned 传输、compile 的详细状态等元数据差异**不会**阻止续训（跨设备续训是
  合法工作流）。

## 数据路径

- **编码器批处理**：`Encoder.encode_clips(B,T,3,H,W)` 一次处理整批；native 后端一次
  `conv3d`，V-JEPA2 一次模型调用。`encode_clip` 委托给它，形状 `(P,D)`、时间轴与因果性
  不变。只实现 `encode_clip` 的第三方编码器会自动退回串行。`encoder.batch_clips` 就是
  真实 batch 大小（含最后一个不满的批次）。
- **世界模型 batch**：`gather_batch` 在主机侧完成时间戳/掩码校验、有效行索引与积分步数
  （`ceil(max(delta)/substep_seconds)`），有效行索引随批次一次搬到计算设备；训练循环用私有 plan 路径，避免每 horizon 一次
  GPU→Python 同步；公开的 `advance/observe/predict` 仍完整校验，传入的 `substeps` 只是
  一致性检查（零 delta 是 no-op，此时不查步数）。
- **解码器 batch**：先在主机侧堆叠 token 行与目标帧，再一次传输、一次 `project`；CUDA 上
  可选 pinned/非阻塞。
- **目标关键帧磁盘缓存**：`decoder_train.target_cache_dir`（默认空 = 关闭）。只存每个分块
  的最后一帧，按来源 SHA-256 + 时间轴 + 采样栅格 + 分块记录 + 输出分辨率寻址，并逐条校验
  像素摘要；写入用唯一临时文件 + 原子替换。**即使全命中缓存，仍会先校验源视频哈希**；
  缓存未命中时会重解码并重建/校验时间轴，命中时不会解码视频。内存中的视频条数由
  `decoder_train.frame_cache_videos` 限制。

## 如何基准测试

### 合成单步基准（包内脚本）

```text
python examples/benchmark.py --out <包外基准目录> [--config <包外 run config>]
                             [--device cuda|cpu] [--batch N] [--decoder-batch N]
                             [--steps N] [--warmup N] [--repeats N]
                             [--encoder-width N] [--grid-side N] [--profile]
```

脚本用**已安装**的包 API 与合成 token/图像，走与世界模型训练相同的、主机侧校验过的 plan
路径，完成整步（zero_grad + forward + backward + 梯度裁剪 + AdamW），世界模型与解码器
各一组。产物（全部写到 `--out`）：

- `benchmark.json`：每步中位数耗时、items/s、各次重复耗时、参数量、CUDA
  allocated/reserved 峰值、TF32 开关、seed、配置、源码签名、`world_compile` 状态，以及
  一行 scope 说明（合成输入、只测计算、不含数据准备）；
- `--profile`：`<case>_trace.json` 与 `<case>_operators.txt`。

它**不包含**：预训练编码器前向、视频 IO、关键帧缓存构建、批次准备、评估与 checkpoint
落盘。所以不要把它的数字与其他用例相加当作端到端收益。

### 编码器与缓存路径

真实 V-JEPA2 编码与缓存请单独测：比较 `encode_clip` 串行与 `encode_clips` 批处理，或直接
计时 `cache` 命令（冷/暖各一次）。这类数字与上面的单步基准是**不同**的用例，单独报告；
把两个隔离用例的加速比相乘或相加都不是有效结论。

### 读数注意

- 合成输入测的是性能，不是模型质量；
- 首次调用包含 cuDNN 算法选择/编译等一次性开销，先 warmup 再取稳态；
- GPU 上请报告 allocated 与 reserved，并说明是否排除桌面/驱动占用；
- FP32 是否启用 TF32、以及 `torch.compile` 是否真的可用，都取决于本机 torch 与后端；
  无法生效时实现会报错或明确记录状态，不会假装启用。本包不依赖自定义 CUDA/Triton 算子。

BF16 与 FP32 的训练轨迹不要求逐位相同；SDPA 采用的具体核也可能不是确定性实现。
保存随机状态和使用相同续训策略，并不保证跨硬件、跨 torch 版本或不同内核的结果逐位相同。
