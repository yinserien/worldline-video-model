# 配置

配置是一个 JSON 文件。顶层只允许 `name`、`data`、`encoder`、`model`、`train`、
`decoder`、`decoder_train` 七个键，未知键会在加载时报错。

- **`data`、`encoder`、`model`、`train` 四个 section 必须存在**，`RunConfig.load`
  直接读取它们；只有 section 内部的字段可以省略并用默认值补齐。
- `decoder` 与 `decoder_train` 是**可选**的（0.2.0 新增）：旧配置没有这两个 section
  也能原样加载，取默认值，世界模型各命令的行为不变。`RunConfig.save` 会把它们写全。
- `name` 可以省略，省略时为 `run`。

生成一份包含全部默认值的配置：

```powershell
$py = "E:\work\wpm_env\Scripts\python.exe"   # 安装了本包的解释器
& $py -c "from wpm_video import RunConfig; RunConfig().save('config.json')"
```

```python
from wpm_video import RunConfig
config = RunConfig.load("config.json")   # str 或 Path
config.validate()                        # load 时已自动调用
```

注意：数据类本身的默认值与 `configs/` 下的预设文件不一定相同。例如
`data.max_horizon_chunks` 的数据类默认值是 **8**，而 `configs/vjepa2_256.json`
里写的是 4。以你手上这份配置文件的实际内容为准。

## 路径规则

- `data.video_dir`、`data.cache_dir`、`--out`、`--checkpoint`、`--state`、`--video`
  等相对路径一律相对**运行时 cwd** 解析，与包安装位置无关。
- `train` 会把生效的配置写入 `<out>/config.json`，把划分写入 `<out>/splits.json`。
- 缓存目录内按编码器与采样设置分子目录，配置改变（fps、分块、图像尺寸、编码器
  revision）会产生新目录，不会误用旧缓存。

## data

| 字段 | 默认值 | 说明 |
|---|---|---|
| `video_dir` | `videos` | 存放 `.mp4` 的目录 |
| `cache_dir` | `cache/tokens` | 编码后 token 缓存目录 |
| `fps` | `4.0` | 采样目标帧率；实际按源帧整数栅格取样，每个采样帧记录其源时间戳 |
| `chunk_frames` | `8` | 每个分块的采样帧数，必须为偶数（编码器 tubelet 成对） |
| `chunk_stride_frames` | `8` | 分块步长，必须 ≥ `chunk_frames`（不允许重叠） |
| `image_size` | `256` | 缩放后的边长（V-JEPA 2 为 256） |
| `context_chunks` | `3` | 一次训练窗口观察的分块数，≥ 2 |
| `window_stride_chunks` | `1` | 训练窗口滑动步长 |
| `horizon_chunks` | `[1, 2]` | 预测目标相对 anchor 的分块数，正整数，≤ `max_horizon_chunks` |
| `max_horizon_chunks` | `8` | horizon 上界（数据类默认值；预设文件里是 4） |
| `train_videos` / `val_videos` | `[]` | 显式划分（文件名不含 `.mp4`）；两者必须同时给出且不重叠 |

时间单位一律为秒。分块的起止时间来自源帧时间戳，预测目标的时间差 =
目标分块 end − anchor 分块 end。

## encoder

| 字段 | 默认值 | 说明 |
|---|---|---|
| `kind` | `vjepa2` | `vjepa2`（真实冻结编码器）或 `native`（随机初始化，仅测试/示例） |
| `model_id` | `facebook/vjepa2-vitl-fpc64-256` | HF 模型 id |
| `revision` | `b3c1679b7c34d3255ef3547f27c7b226aefab26f` | 固定 revision |
| `device` | `cuda` | 编码器所在设备，`cuda` / `cpu` / `auto` |
| `batch_clips` | `4` | 缓存时一次处理的片段数 |
| `allow_native_fallback` | `false` | 仅测试用；真实 run 不会静默退回随机权重 |

`kind = "native"` 时 `model_id` / `revision` 被忽略（缓存键按 native 处理），
并且需要命令行加 `--allow-native`。

## model

| 字段 | 默认值 | 约束 |
|---|---|---|
| `d_world` | `256` | 正整数，≤ 编码器宽度，能被 `heads` 整除 |
| `d_hidden` | `512` | MLP 隐层宽度 |
| `slots` | `64` | 持久槽位数，必须是完全平方数（8×8 锚点网格） |
| `heads` | `4` | 注意力头数，≥ 1 |
| `projection_seed` | `20261006` | 固定随机投影种子（不参与训练） |
| `a_max` | `1.0` | 加速度上界，有限非负数 |
| `damping_init` | `0.5` | 阻尼系数初值，(0, 2] |
| `substep_seconds` | `0.25` | 积分步长，必须 ≤ 0.5 |
| `max_substeps` | `256` | 单次 `advance` 的最大积分步数；超限时报错 |
| `logvar_min` / `logvar_max` | `-8.0` / `6.0` | 对数方差的钳制范围 |
| `anchor_bandwidth` | `2.0` | 位置寻址带宽，有限非负数 |
| `velocity_write` | `0.5` | 写入时速度项的固定耦合系数 |
| `kl_beta` | `0.01` | rate 项权重 |
| `w_horizon` / `w_prior` / `w_present` | `1.0` / `0.5` / `0.5` | 三项损失权重，至少一个为正 |

以上字段在模型构造时检查；`model.config` 保存的就是这一组生效值
（例如 `model.config.max_substeps`）。

## train

| 字段 | 默认值 | 说明 |
|---|---|---|
| `seed` | `0` | 训练随机种子；在 `train` 开始时生效（采样器、随机数状态） |
| `batch_windows` | `8` | 每步窗口数 |
| `learning_rate` | `0.0003` | AdamW 学习率 |
| `weight_decay` | `0.0` | AdamW 权重衰减 |
| `max_steps` | `3000` | **总目标步数**；续训时要大于 checkpoint 的 step |
| `grad_clip_norm` | `5.0` | 梯度裁剪范数 |
| `eval_interval` | `250` | 验证间隔（步） |
| `eval_batches` | `8` | 每次验证的 batch 数 |
| `max_wall_seconds` | `1800.0` | 训练墙钟上限 |
| `device` | `cuda` | 训练与推理设备，`cuda` / `cpu` / `auto` |
| `resume_from` | `""` | 保留字段，当前命令行不使用；续训用 `--resume`（见 [commands.md](commands.md)） |
| `log_interval` | `25` | 训练日志间隔（步） |

`seed` 只在 `train` 开始时调用 `set_determinism`，**不会**固定此前新建模型的参数
初始化；API 需要可复现时先自己调用 `set_determinism(config.train.seed)` 再
`build_model`。

`encoder.device` 与 `train.device` 各自生效：缓存用前者，训练/评价/推理用后者；
`predict` 与 `query` 使用 `train.device`。

## decoder

可选 RGB 解码器的**架构**，只在 `train-decoder` 与传了 `--decoder-checkpoint` 的
`predict`/`query` 里生效；不训练解码器时可以整段忽略。详见
[decoder.md](decoder.md)。

| 字段 | 默认值 | 约束 |
|---|---|---|
| `image_size` | `128` | 输出边长，`[32, 2048]` 且为 8 的倍数；另需是 patch 网格边长的 2 的幂倍（在构建时检查） |
| `base_channels` | `128` | `[1, 2048]` |
| `channel_multipliers` | `[1, 2, 2]` | 非空，每项为 `[1, 64]` 的整数 |
| `stem_blocks` | `2` | 整数 `>= 1`，网格分辨率上的卷积层数 |
| `blocks_per_stage` | `1` | 整数 `>= 1`，每次上采样后的卷积层数 |

## decoder_train

解码器的优化设置（`train-decoder` 使用），与世界模型的 `train` 相互独立。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `seed` | `0` | 非负整数；在**构建解码器之前**生效，保证 initial checkpoint 与续训一致 |
| `batch_windows` | `8` | 每步窗口数（训练与验证共用） |
| `learning_rate` | `0.0003` | AdamW 学习率，正有限 |
| `weight_decay` | `0.0` | 有限非负 |
| `max_steps` | `2000` | **总目标步数**；续训时要大于 checkpoint 的 step |
| `grad_clip_norm` | `5.0` | 梯度裁剪范数，正有限 |
| `eval_interval` | `100` | 验证间隔（步） |
| `eval_batches` | `4` | 每次验证的 batch 数 |
| `log_interval` | `25` | 日志间隔（步） |
| `max_wall_seconds` | `1800.0` | 墙钟上限；到点会先对最终 checkpoint 补一次验证 |
| `l1_weight` | `1.0` | 像素 L1 权重，有限非负 |
| `edge_weight` | `0.1` | 边缘（有限差分 L1）权重，`0` 关闭；与 `l1_weight` 不能同时为 0 |
| `frame_cache_videos` | `2` | 目标帧按视频惰性解码，最多同时在内存里保留几条 |

世界模型训练写出的 `config.json` 里包含这两段，因此把同一份配置交给
`train-decoder` 时，采样参数与编码器身份天然一致。

## 检查点与状态

世界模型的 checkpoint（`initial.pt` / `best.pt` / `final.pt`）保存：模型与配置、投影与
标准化 buffer、优化器、步数、采样器与 CPU/CUDA 随机状态、训练 provenance。

解码器的 checkpoint 名字相同但内容不同（`kind="rgb_decoder"`）：解码器配置与架构、
**world 投影 buffer 的 SHA-256 指纹**、编码器身份、采样参数、目标帧语义与分辨率、
优化器、步数、两个采样生成器与 CPU/CUDA 随机状态。给 `--decoder-checkpoint` 传世界模型
checkpoint 会被明确拒绝。见 [decoder.md](decoder.md)。

- 用 `--resume <checkpoint>` 继续训练：恢复优化器与随机状态，沿用配置里的数据划分，
  `max_steps` 为总目标步数。`initial.pt` 只在新训练时写出，`best.pt` 只在验证指标
  改善时写出。
- 独立状态文件（`world_state.pt`）保存持久状态（slots、变化率、float64 时钟、
  步数），只能由 `predict`/`save` 产生，可用 `query` 或 Python API 读取，
  不需要视频或编码器。

## 评价指标单位

- `nll_bits_per_dim`：未来表征的负对数似然，**bits per dimension**（在 batch、
  空间 patch、d_world 上取均值）。`train_log.jsonl` 里的键是
  `nll_bits_per_dim_horizon`（各 horizon 汇总）、`nll_h<N>_bits_per_dim`（逐 horizon）、
  `prior_nll_bits_per_dim`、`present_nll_bits_per_dim`；`metrics.json` 里是
  `per_horizon` / `per_anchor_horizon` 下的 `nll_bits_per_dim`。
- `kl_bits_per_dim` / `kl_bits_per_event`：prior 与 posterior 之间的 KL，前者是
  每个维度的均值，后者是单个分块事件的总 bits。只在 `train_log.jsonl` 里。
- `mse`：表征空间的均方误差（同样的维度上取均值）；`train_log.jsonl` 与
  `metrics.json` 里都叫 `mse`（`train_log.jsonl` 里是各 horizon 的汇总值）。
- `delta_seconds`：该组评价样本的真实时间差均值，单位秒，出现在 `metrics.json`。
- 时间字段（`state_time_seconds`、`target_time_seconds`、`*_end_seconds`）一律为
  float64 秒。

## 可选 PCA

`predict` 的输出里 `latent_pca.png` 是**可选**可视化：仅用训练集 latent 拟合，
绝不使用推理视频。当 checkpoint 没有训练划分信息，或本机没有对应的 token 缓存时，
会跳过并在 `pca_meta.json` 记录 `status` 与 `reason`；其余产物不受影响。

## 可选 RGB 解码

`--decoder-checkpoint` 是**可选**的：不传时 `predict`/`query` 只产出 latent 与既有的
非 RGB 图表，`predict_summary.json` 的 `decoded_rgb` 为 `null`。传了以后才会多出
`decoded_*.png` 等关键帧产物；解码器与 world checkpoint 的身份不匹配时，在写出任何图片
之前就会报错。字段、指纹与产物见 [decoder.md](decoder.md) 与
[commands.md](commands.md#predict)。
