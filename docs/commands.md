# 命令行

入口：`python -m wpm_video`（安装后控制台脚本 `wpm-video` 等价，但需要环境已激活或
`Scripts` 在 PATH 上）。所有相对路径相对运行时 cwd 解析，`--config` 是必需的。

下面各命令的 `text` 代码块是**语法形式**（`<...>` 是占位符，不是可执行命令）；
可直接复制的完整流程见 [README](../README.md#用自己的视频)。`powershell` 代码块里的
`$py` 指安装了本包的解释器（Windows 上如 `E:\work\wpm_env\Scripts\python.exe`），
调用时必须有 `&` 运算符。

```powershell
$py = "E:\work\wpm_env\Scripts\python.exe"
& $py -m wpm_video --help
& $py -m wpm_video predict --help
```

通用可选参数：`--out`、`--allow-native`（允许随机初始化的 native 编码器，仅用于
测试与离线示例）。

`--out` 的类型按命令区分：**`cache` 与 `provenance` 的 `--out` 是 JSON 文件路径**，
其余命令的 `--out` 是目录。

## cache

编码每个视频的因果分块并写缓存。编码器只读取该分块内的帧。

```text
wpm-video cache --config <config> [--out <manifest.json>]
```

产物：缓存目录 `<cache_dir>/<encoder-tag>/<clip>.pt`（tokens + 每个分块的真实
时间戳），以及 manifest JSON（默认 `<cache_dir>/manifest.json`，含源视频 SHA-256、
来源 identity、编码器 revision）。若 `video_dir` 下存在 `sources.json`，会把其中
声明的公开来源与文件 bytes 逐字节核对后再写入记录。

## train

```text
wpm-video train --config <config> --out <dir>
                [--train-videos A B ...] [--val-videos C D ...] [--val-count N]
                [--resume <checkpoint>]
```

划分优先级：

1. 命令行同时给出 `--train-videos` 与 `--val-videos`；
2. 否则用配置文件里的 `data.train_videos` 与 `data.val_videos`（两者都有时）；
3. 否则按文件名顺序自动留出 `--val-count N` 条作验证集（默认 1）。

训练与验证不得共享同一文件或同一来源 identity，否则报错。生效的划分写入
`<out>/splits.json`（含 `source` 字段说明来自哪一级）。

- `--resume <checkpoint>` 从 checkpoint 继续：恢复优化器与 CPU/CUDA 随机状态，
  沿用配置里的划分，`max_steps` 是总目标步数（要大于 checkpoint 的 step）。
  续训请沿用上一个 run 的 `config.json` 以保留真实划分。
- 产物：`config.json`、`splits.json`、`train_log.jsonl`（每 `log_interval` 步一行）、
  `train_summary.json`，以及 checkpoint：
  - `initial.pt`：**仅新训练**（非 `--resume`）时写出，step 0；
  - `best.pt`：**仅验证指标改善时**写出；
  - `final.pt`：训练结束（达到 `max_steps` 或墙钟上限）时写出。

## train-decoder

可选：训练 RGB 解码器。编码器与世界模型保持冻结，输入是缓存 token 的投影 latent，目标是
同一分块在源视频里的**最后采样帧**。

```text
wpm-video train-decoder --config <config> --checkpoint <world 的 checkpoint> --out <dir>
                        [--resume <decoder checkpoint>]
```

- 划分来自 world checkpoint 的 provenance，**不能**用命令行指定；checkpoint 必须由本包的
  `train` 写出。train/val 共享文件或来源 identity 时报错。
- 训练前会核对调用方 config 的编码器身份/采样参数与 checkpoint 记录的配置一致、token
  缓存由该编码器写出、源视频与缓存逐字节一致且分块时间戳完全对齐；不通过就不开始训练。
- 只用训练集拟合"常量帧参考"，不在验证集上拟合任何统计量。
- `--resume` 恢复优化器、步数、采样器与分块生成器、CPU/CUDA 随机状态；**checkpoint 的
  架构（含输出分辨率）是权威**，与 config 的 `decoder` 段不一致会报错。
- 产物：`initial.pt`/`best.pt`/`final.pt`（按留出 L1）、`train_log.jsonl`、
  `train_summary.json`（含 `l1`/`mse`/`psnr_db`、`constant_frame_reference`、`per_video`）、
  `config.json` 与 `splits.json`（写实际生效的架构与划分）。
- 细节与指标定义见 [decoder.md](decoder.md)。

## eval

```text
wpm-video eval --config <config> --checkpoint <run>/best.pt --out <dir> [--batches N]
               [--checkpoints A.pt B.pt ...]
```

默认评估同一 run 的 `initial/best/final`（存在哪个评哪个），使用相同验证窗口与三条
基线（persistence、train_mean、linear_extrapolation），每行同时给出 NLL 与 MSE。

需要 checkpoint 带训练划分 provenance，**并且本机要有该划分的 train 与 val token
缓存**（基线统计量用 train，评分用 val）；否则请改用 `predict`。产物：
`metrics.json`（`checkpoints`、`baselines`、`baseline_summary`、
`learning_check_initial_vs_best`、`validation_provenance`、`wall_seconds`）。

## predict

```text
wpm-video predict --config <config> --checkpoint <checkpoint> --video <clip.mp4>
                  --out <dir> [--prefix-chunks N] [--state <world_state.pt>]
                  [--decoder-checkpoint <decoder.pt>]
```

- 只需要 checkpoint、编码器配置和视频；不需要训练数据或缓存，可跨机器使用。
- `--state` 从已保存状态继续：已经被该状态观察过的分块会跳过，不会让时间倒退。
- 视频短于 `--prefix-chunks` 时按实际分块数处理，并在 summary 记录
  `prefix_clamped` 与 `chunks_in_video`。
- `--decoder-checkpoint` **可选**：给出后额外把预测 latent 与真值未来 latent 解码成关键帧
  图片。兼容性在写任何图之前检查，不匹配直接报错。
- 产物：
  - `world_state.pt`：持久状态（slots、速度、float64 时钟、步数）
  - `future_latents.pt`：`{"predictions": {horizon: {...}}, "meta": {...}}`，
    每个 horizon 只保存**张量**字段 `mu`、`sigma`，有真实未来分块时另有 `target`
  - `predict_summary.json`：视频与来源哈希、前缀与钳制信息、状态时间；每个 horizon
    的非张量字段都在这里（`horizon_chunks`、`delta_seconds`、`target_time_seconds`、
    `scored`、`target_start_seconds`、`target_end_seconds`，有真值时还有
    `nll_bits_per_dim` 与 `mse`）；`decoded_rgb` 记录解码产物与时间语义，未启用时为 `null`
  - `frames.png`：观测前缀帧与真实未来帧（参考用，不生成新画面）
  - `uncertainty.png`：预测 σ 与实测误差
  - `pca_meta.json`（以及可用时的 `latent_pca.png`）：见下
  - 仅当传了 `--decoder-checkpoint`：
    - `decoded_h<horizon>.png`：**预测 mu** 解码出的关键帧
    - `target_reconstruction_h<horizon>.png`：解码器作用在**真值未来 latent** 上（单独
      评价解码器）
    - `decoded_predictions.pt` / `decoded_targets.pt`：`float32 (3, S, S)`，`[0, 1]`
    - `decoded_summary.json`：每张图的时间/horizon 元数据
    - `decoded_frames.png`：观测前缀+真值帧 / 预测解码 / 真值 latent 重建 的三行对照图

  解码帧是**彼此独立的关键帧**（按 horizon 排列），不是连续视频；时间元数据同时给出
  `chunk_end_seconds` 与 `target_frame_timestamp_seconds`（后者是分块内最后被采样帧的真实
  时间，即分块结束时间减去一个源帧周期）。

PCA 说明：`latent_pca.png` 只用训练集 latent 拟合。checkpoint 无训练划分信息或
本机无对应 token 缓存时跳过，`pca_meta.json` 记录 `status="skipped"` 与 `reason`；
其余产物照常生成。

## query

```text
wpm-video query --config <config> --checkpoint <checkpoint> --state <world_state.pt>
                --deltas <秒> [<秒> ...] [--out <dir>] [--decoder-checkpoint <decoder.pt>]
```

从状态出发预测若干未来时刻，不需要视频与编码器；加了 `--decoder-checkpoint` 也不需要
（解码只吃 latent），但必须有 `--out` 才能写图。产物：

- `state_query.json`：`state_path`、`state_time_seconds`、`state_step`，以及
  `queries`——以 Δ 秒字符串为键，每项含 `target_time_seconds`、`mean_sigma`、
  `mu_std`；`decoded_rgb` 为解码元数据或 `null`；
- `state_query_latents.pt`：以 Δ 秒字符串为键，每项含张量 `mu`、`sigma`；
- 仅当传了 `--decoder-checkpoint`：`decoded_d<Δ>s.png`、`state_query_decoded.pt`、
  `decoded_summary.json`。查询没有视频，因此 `source_fps` 与
  `target_frame_timestamp_seconds` 记为 `null`，并在 `timestamp_basis` 说明像素时间未知
  ——分块结束时间只是查询锚点，不会被当成帧的拍摄时间。

## provenance

```text
wpm-video provenance --config <config> [--revision <rev>] [--snapshot <dir>]
```

把 `video_dir` 中每个片段映射到下载快照中的原文件，**逐字节比对**后才写入
`--out`（默认 `video_dir/sources.json`，是一个 JSON 文件）；不一致或找不到对应
文件的片段不声明公开来源，列在 `skipped` 中。

## selfcheck

```text
wpm-video selfcheck --config <config>
```

运行架构契约检查（空间寻址、时间契约、确定性、单位、状态序列化等），数秒完成，
不需要下载或 GPU。
