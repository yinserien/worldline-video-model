# 可选 RGB 解码器

解码器把**一个分块**的投影后世界 latent `(P, d_world)` 还原成**一帧 RGB 关键帧**
`(3, S, S)`，取值 `[0, 1]`。它回答的是"模型此刻认为这个分块最后被采样到的那一帧长
什么样"，不是视频生成：

- 每个输出对应分块内**最后被采样的那一帧**（`end_frame - 1`），不是分块内所有帧；
- 按 horizon 排列的若干张图是**彼此独立的关键帧**，不是一段连续视频，也不是高帧率
  序列；文件命名与 JSON 元数据都按这个语义标注；
- 输入是冻结编码器 + 随机投影的 latent，信息本身有损。**真值 latent 的重建也会发虚、
  丢纹理**；不要把输出称为"生成画面"，除非你提供了自己的解码器 checkpoint 并报告了
  它在留出视频上的指标。

模型本身完全可以不用它：不传 `--decoder-checkpoint` 时，`train`/`eval`/`cache`/
`predict`/`query` 继续使用原有 latent 推理流程，不输出 RGB 图片；已有 latent 产物保持
兼容，汇总 JSON 只增加 `decoded_rgb: null` 元数据。

## 位置与职责

```
src/wpm_video/decoder/
    base.py              RGBDecoder 基类：组件契约与共享几何校验
    registry.py          register_decoder / available_decoders / build_decoder 按 kind 分发
    architectures/conv.py 内置 conv 架构（kind="conv"），导入即注册
    model.py             checkpoint 格式（schema 1/2）与 save/load；旧的 LatentRGBDecoder、
                         build_decoder、group_count 等导入路径仍然有效
    compat.py            投影指纹、编码器/采样身份、兼容性检查
    targets.py           目标帧读取（按视频有界惰性缓存）与时间轴对齐校验
    train.py             独立的解码器训练/评价/产物
    render.py            latent -> PNG 与张量产物的共享渲染助手
```

动力学（`model.py`、`world_state.py`、`train.py`）**不 import** 这个子包；解码只发生在
推理末尾，作用于已经算好的 latent。

解码器是**可替换组件**：`config.decoder.kind` 选择一个已注册的架构，内置的
`"conv"`（就是下面这个卷积上采样器）是默认值。要接入第二种架构，见
[decoder_components.md](decoder_components.md)；本文余下的内容描述的都是这个默认架构。

## 架构与参数量

```
(B, P, d_world) -> 按 (gh, gw) 重排 -> 加位置嵌入 -> stem 卷积（网格分辨率）
                -> log2(S / gh) 次 x2 最近邻上采样 + 卷积块 -> 3 通道 + sigmoid -> (B, 3, S, S)
```

`S = decoder.image_size`，`gh` 为 patch 网格边长，`S / gh` 必须是 2 的幂（否则报错，
绝不做隐式重采样）。默认配置 `DecoderConfig()`：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `kind` | `"conv"` | 组件架构；`"conv"` 是内置且默认的这一个，其他取值见 [decoder_components.md](decoder_components.md) |
| `options` | `{}` | 该架构自己的设置；**conv 必须为空**（conv 用下面四个字段配置） |
| `image_size` | `128` | 输出边长，必须是 8 的倍数；conv 进一步要求它是 patch 网格边长的 2 的幂倍（自定义架构的几何约束由架构自己定） |
| `base_channels` | `128` | stem 与第一级通道数（仅 conv） |
| `channel_multipliers` | `[1, 2, 2]` | 各级通道倍率；级数由 `image_size / 网格边长` 决定，超出的级复用最后一项（仅 conv） |
| `stem_blocks` | `2` | 网格分辨率上的卷积层数（仅 conv） |
| `blocks_per_stage` | `1` | 每次上采样后的卷积层数（仅 conv） |

`kind` 与 `options` 追加在原有字段之后，位置参数写法与旧配置文件的含义都不变。

参数量（16x16 patch 网格、128 像素输出，含位置嵌入）：

| `d_world` | 参数量 |
|---|---|
| 256 | 1,549,699 |
| 512 | 1,910,147 |
| 768 | 2,270,595 |
| 1024 | 2,631,043 |

## 训练

```text
wpm-video train-decoder --config <config> --checkpoint <world 的 best.pt> --out <dir>
                        [--resume <decoder checkpoint>]
```

- 编码器、世界模型、投影全部**冻结**：输入是 `model.project(缓存 tokens)`，目标是同一
  分块的**最后采样帧** RGB（直接从源视频解码，`targets.py`）。每个 batch 先在主机侧堆叠
  token 行与目标帧，再一次传输、一次 `project`。
- 划分来自 **world checkpoint 的 provenance**（`--checkpoint` 必须由本包的 `train` 写
  出）；train/val 之间按文件哈希与来源 identity 拒绝重叠。
- 校验分两段，读日志时注意区别：
  - **第一个梯度步之前（eager）**：调用方 config 的编码器身份与采样参数要和 world
    checkpoint 记录的配置一致、每个 token 缓存要由该编码器写出且记录完整（编码器、
    宽度、patch 数、分块列表、token 行数）；
  - **首次用到某条源视频时（lazy）**：才比对该文件的 SHA-256，并重新解码核对采样栅格、
    分块边界与时间戳。视频按需逐条解码（有界 LRU），不会在开训前把所有素材都读一遍；
    被替换过的片段会在它第一次被用作目标之前报错。
- 只用训练分块拟合的统计量是"常量帧参考"（均值帧），**不会在验证集上拟合任何东西**。
- 损失：L1 + 小权重边缘项（有限差分 L1，`decoder_train.edge_weight`，置 0 可关闭）。
  基线版本不使用 GAN / diffusion / 感知网络。
- 指标（留出视频）：`l1`、`mse`（`[0,1]` 像素均值）与 `psnr_db = 10*log10(1/mse)`，
  另有 `constant_frame_reference` 作对照；逐视频结果在 `per_video`。
- 产物：`initial.pt` / `best.pt` / `final.pt`（按留出 L1 取最优）、`train_log.jsonl`、
  `train_summary.json`、`config.json`、`splits.json`。`config.json` 写的是**实际生效**的
  架构与划分。
- `--resume` 恢复优化器、步数、采样器/分块生成器与 CPU/CUDA 随机状态；**checkpoint 的
  架构是权威**，config 与它不一致会直接报错（否则会记录一套配置却训练另一个网络）。
  比对的架构是**规范化记录**：组件 `kind`、`image_size` 与自定义 `options` 都包含在内，
  任何差异都在第一个梯度步之前拒绝。续训还会比对 `performance` 的语义项（精度/注意力核/
  fused/compile），不一致时报错；decoder 始终不编译，世界模型的 `compile_scope` 不会启用
  decoder 编译。旧版本写的 checkpoint 没有该记录，视为参考策略。
- `train_summary.json` 的 `decoder` 段按**组件**记录：`kind`、`architecture`（含 options）、
  `parameters`、`image_size`、`grid`、`d_world`——不再假定 conv 的 `base_channels` /
  `channel_multipliers` 存在。
- checkpoint 现在写 **schema 2**（记录 `kind`/`options`）；v0.4.0 及更早的 **schema 1**
  仍可读取、比对与续训，它只可能描述内置 conv，因此按"旧 conv"解释而不是按形状猜测。
  未知/未来 schema、未知 kind、被改写的元数据都会在产生像素或梯度之前报错。详见
  [decoder_components.md](decoder_components.md#checkpoint-格式与兼容)。

- 可选的目标关键帧磁盘缓存：`decoder_train.target_cache_dir`（默认空 = 关闭）。只保存每个
  分块**最后一帧**（不保存整段视频），按来源 SHA-256、时间轴 schema、采样帧率与栅格、分块
  记录（index/起止帧/起止时间）与输出分辨率寻址，条目内另有像素+时间戳+分块记录的摘要；
  读取时逐条校验，写入用唯一临时文件 + 原子替换（并发写同一路径时先到者生效，内容按构造
  相同）。**即使全部命中缓存，首次用到某条视频时仍会校验源文件 SHA-256**；缓存未命中时会
  重新解码并重建/校验时间轴。视频在内存中最多保留 `decoder_train.frame_cache_videos` 条。
  首次构建缓存要付一次解码成本，之后的运行省掉解码；`train_summary.json` 的
  `decoded_frame_cache` 记录解码条数、命中/未命中、写入条数与字节数。

`decoder_train` 字段见 [configuration.md](configuration.md#decoder_train)。

## 推理

```text
wpm-video predict --config <config> --checkpoint <world checkpoint> --video <clip.mp4> \
                  --out <dir> [--decoder-checkpoint <decoder.pt>]
wpm-video query   --config <config> --checkpoint <world checkpoint> --state <state.pt> \
                  --deltas 1 2 4 --out <dir> [--decoder-checkpoint <decoder.pt>]
```

- 解码前先做兼容性检查：投影 buffer 指纹、`d_world`、patch 网格、编码器身份、采样参数、
  目标帧语义与分辨率，以及**组件 kind 与规范化架构**。**维度相同但投影不同、或架构不同，
  都会被拒绝**；不兼容时不会写出任何一张图。加载自定义架构的 checkpoint 需要该进程已注册
  对应 kind（见 [decoder_components.md](decoder_components.md#注册的生命周期)）。
  采样身份只包含 `timeline_schema`、`fps`、`chunk_frames`、`chunk_stride_frames`、
  `image_size`、`patches`：`context_chunks` 属于世界模型的观测窗口，只作为 provenance
  记录，改它不会让已有解码器失效。
- 设备语义：`decode_latents(decoder, latents, device)` 会把解码器与 latent 都放到
  `device` 上再推理（`load_decoder` 默认加载到 CPU），并恢复调用前的 train/eval 模式；
  解码器会留在该设备上。CLI 已经按 `train.device` 处理，Python API 里显式写
  `decoder = decoder.to(device).eval()` 更清楚。
- 精度：解码器训练同样受 `performance.precision` 控制，但重建损失与 `l1`/`mse`/`psnr`
  始终在 FP32 计算；`predict` 的默认路径保留注意力权重（reference 读），只有内部无权重
  路径使用配置的 SDPA。加速项与数值契约见 [performance.md](performance.md)。
- `predict` 的产物：
  - `decoded_h<horizon>.png`：**预测 mu** 解码出的关键帧；`decoded_predictions.pt` 存
    `float32 (3, S, S)`、`decoded_summary.json` 存时间元数据；
  - `target_reconstruction_h<horizon>.png` / `decoded_targets.pt`：把解码器作用在
    **真值未来 latent** 上，用来单独评价解码器（把预测误差排除在外）；
  - `decoded_frames.png`：三行对照图（观测前缀与真值帧 / 预测解码 / 真值 latent 重建）。
- `query` 只解码状态查询得到的 `mu`，**不需要视频与编码器**；因为没有任何视频信息，
  `target_frame_timestamp_seconds` 与 `source_fps` 记为 `null`，并在 `timestamp_basis`
  里说明"分块结束时间是查询锚点，不是像素时间"。
- `predict` 知道源 fps，因此时间戳是**真实的**：解码帧是分块内最后被采样的一帧，时间 =
  分块 `end_seconds` 减去一个源帧周期，元数据里同时给出 `chunk_end_seconds`、
  `target_frame_timestamp_seconds` 与 `source_fps`。
- 默认产物保持兼容：**不传 `--decoder-checkpoint` 时 latent 产物
  （`future_latents.pt`、`state_query_latents.pt`、`world_state.pt`）与非 RGB 图表的
  内容不变**，`predict_summary.json` / `state_query.json` 只是新增了 `decoded_rgb`
  这一段附加元数据（未启用时为 `null`），既有的键与数值不变。
- **未来真值不会进入状态或预测像素**：解码只在所有 latent 算完之后进行，只读取已经算好的
  `mu`（预测）与 `target`（真值），且只写文件，不回流到状态、预测或任何其他图像。真值
  未来 latent 会被**单独**解码成 `target_reconstruction_*.png`，那是明确标注的参考产物
  （用于把解码误差和预测误差分开），不会被当成预测结果。
- 查询 Δ 保持 float64：查询键、`delta_seconds` 与 `target_time_seconds` 都是调用方给出的
  精确时间，`1.0` 与 `1.000000001` 会得到两个条目、两张文件名不同的图（只有激活值是
  float32）。

## 已知限制

- 重建来自有损的冻结表征 + 随机投影，默认基线只做逐像素与边缘监督，画面偏糊、纹理缺失
  是预期结果；
- 解码器只见过"真实分块 latent"，作用在"预测 mu"上时误差 = 预测误差 + 解码误差，两者在
  产物里分开报告；
- 输出分辨率是配置项，但不会超过训练时使用的分辨率所带来的细节上限；
- 默认基线只有逐像素与边缘监督：不含 GAN / diffusion / 感知损失，也没有自定义算子。
  想要更锐利的画面需要换架构或换目标，属于**后续要按对照协议测量**的实验，见
  [decoder_components.md](decoder_components.md#对照实验协议)；
- 单卡 CPU/单 GPU 的线性训练路径，没有分布式；混合精度由 `performance.precision` 控制
  （见 [performance.md](performance.md)），decoder 不做 `torch.compile`。
