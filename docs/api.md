# Python API

从包顶层导入（`Path` 供下面需要 `Path` 的 `out_dir` 使用）：

```python
from pathlib import Path

from wpm_video import RunConfig, VideoWorldModel, WorldState, build_encoder
```

## 路径参数类型

| 参数 | 类型 |
|---|---|
| `RunConfig.load(path)` / `RunConfig.save(path)` | `str` 或 `Path` |
| `WorldState.load(path, ...)` / `state.save(path)` | `str` 或 `Path` |
| `probe_video` / `read_frames` / `list_videos` / `load_source_manifest` / `build_source_manifest` | `str` 或 `Path` |
| `cache_video_tokens(video, ..., out_dir)` | 二者皆可 |
| `demo(..., video, ..., out_dir)` | 二者皆可 |
| `query_saved_state(model, state_path, ..., out_dir=...)` | `state_path` 二者皆可；**`out_dir` 必须是 `Path`** |
| `train(config, model, train_set, val_set, out_dir, ...)` | **`out_dir` 必须是 `Path`** |

`train` 与 `query_saved_state` 直接对 `out_dir` 调用 `mkdir`，传 `str` 会抛
`AttributeError`。CLI 里这两处传的就是 `Path`。

## 配置

```python
config = RunConfig.load("config.json")   # 也接受 Path
config.validate()                        # load 时已自动调用
config.save("config.json")               # 写回（含默认值）
config.to_dict()                         # 嵌套 dict
```

字段与约束见 [configuration.md](configuration.md)。

## 编码器

```python
encoder = build_encoder(config.encoder, device, allow_native=False)
encoder.kind, encoder.d_model, encoder.patches   # 后端、token 宽度、空间 patch 数
tokens = encoder.encode_clip(clip)               # (T,3,H,W) uint8 -> (P, d_model)
tokens = tokens.to(device).unsqueeze(0)          # -> (1, P, d_model)，observe 需要这个形状
```

`vjepa2` 为冻结的真实编码器（推理在 `torch.no_grad` 下）；`native` 为随机初始化，
需要 `allow_native=True`，只用于测试与离线示例。真实 run 不会静默退回 native。

## 视频与分块

```python
from wpm_video import probe_video, read_frames, build_chunks, list_videos

info = probe_video("clips/a.mp4")            # fps、帧数、时长、sha256、source_id
frames, timestamps = read_frames("clips/a.mp4", config.data.fps, config.data.image_size)
chunks = build_chunks(info, config.data, timestamps)
```

- `frames`：`(T, 3, H, W)` uint8；`timestamps`：`(T,)` float64 秒（源时间轴）。
- `chunks`：每个分块带 `start_frame/end_frame/start_seconds/end_seconds`，
  时间取自源帧时间戳，相邻分块不重叠。

## 模型与状态

```python
model, payload = VideoWorldModel.from_checkpoint("best.pt")   # payload 含配置与 provenance
model = model.to(device).eval()
state = model.initial_state(batch, device)                    # 持久状态，初始为 learned 初值
```

`payload` 是 checkpoint 字典：`model_config`、`patches`、`chunk_seconds`、
`state_dict` 必定存在；`provenance`、`step`、`optimizer`、`history`、`best` 只有在
checkpoint 由 `train` 写出时才存在，用 `VideoWorldModel.save(...)` 保存的裸
checkpoint 没有 `provenance`（读它之前先 `payload.get("provenance")`）。

`WorldState` 字段：

| 字段 | 形状 / 类型 | 含义 |
|---|---|---|
| `slots` | `(B, slots, d_world)` | 内容寻址的空间锚点 |
| `velocity` | `(B, slots, d_world)` | 显式变化率 |
| `time` | `(B,)` float64 | 源时间轴上的秒 |
| `step` | `(B,)` int64 | 已观测分块数 |

方法：`clone()`、`detach()`、`to(device=, dtype=)`、`select(index)`、`save(path)`、
`WorldState.load(path, device=, dtype=)`。时钟始终为 float64；`dtype` 只影响
slots/velocity。

## 观测、推进与预测

```python
state, diagnostics = model.observe(state, tokens, end_seconds, sample=None, generator=None)
state = model.advance(state, delta_seconds)          # 不读观测
mu, logvar, advanced, attention = model.predict(state, delta_seconds)
```

- `observe`：`tokens` 形状 `(B, P, d_model)`，`P` 必须等于 `encoder.patches`；
  先把状态推进到该分块的真实结束时间，再用 prior/posterior 计算 innovation 并写入
  状态。调用方负责 `encode_clip(...).unsqueeze(0)`。`end_seconds` 必须严格递增；
  时间倒退、重复或非有限值会报错。`sample=None` 跟随 `model.training`：`eval()` 时
  用后验均值（确定性），训练时采样；传入 `generator` 可固定随机源。
- `advance`：只改变状态，不读任何观测；`delta_seconds` 可为标量或每行一个
  （`(B,)` 张量），必须非负、有限，且不超过 `model.config.max_substeps` 个积分步。
  `delta_seconds == 0` 返回等值副本。
- `predict`：返回 `mu`、`logvar`（形状 `(B, P, d_world)`）、推进后的状态与注意力。
  `logvar` 是**对数方差**，范围由 `model.config.logvar_min/logvar_max` 限制；
  `sigma = exp(0.5 * logvar)`。`observe` 与 `predict` 都接受 batched 状态。

## 流式与查询

```python
from wpm_video import stream_chunks, predict_at, predict_future, query_saved_state, demo

state, log = stream_chunks(model, encoder, frames, chunks[:3], device, state=None)
futures = predict_at(model, state, [1.0, 2.0])          # 只用状态与 Δ 秒
futures[2.0]["mu"], futures[2.0]["sigma"], futures[2.0]["target_time_seconds"]
result = demo(model, encoder, config, "clips/a.mp4", device, "runs/predict",
              prefix_chunks=3, state_path=None)          # 等价于 CLI predict
saved = query_saved_state(model, "state.pt", [1.0, 4.0], device, Path("runs/query"))
```

- `stream_chunks` 与 `predict_at` 只处理 **batch 1** 的状态（内部取 `[0]`），并为每个
  分块产出一条日志：`skipped` 是被状态覆盖时的说明字符串（`"already covered by the
  state"`），正常观察时为 `False`。
- `predict_at` 返回以 Δ 秒为键的字典，每项含张量 `mu`、`sigma`（形状
  `(P, d_world)`），以及 `delta_seconds`、`target_time_seconds`、`scored`。
- `demo` 把张量写入 `future_latents.pt`（结构 `{"predictions": {horizon: {张量}},
  "meta": {...}}`，每个 horizon 只含 `mu`/`sigma`，有真值时含 `target`），把
  horizon、delta、时间戳与评分写入 `predict_summary.json`。
- `predict_at` / `query_saved_state` 不需要视频与编码器。

## 训练与评价

```python
from pathlib import Path
from wpm_video import (TokenDataset, build_encoder, build_model, evaluate_model,
                       fit_projection, set_determinism, train)

train_set = TokenDataset(config.data, config.data.cache_dir, config.encoder, "train")
val_set = TokenDataset(config.data, config.data.cache_dir, config.encoder, "val")

set_determinism(config.train.seed)          # 需要可复现时，在 build_model 之前调用
encoder = build_encoder(config.encoder, device)
model = build_model(config, encoder, device)
fit_projection(model, train_set)            # 在训练 token 上拟合标准化（只用 train）
summary = train(config, model, train_set, val_set, Path("runs/run1"), device,
                provenance={"splits": {"train": [...], "val": [...]}})
metrics = evaluate_model(model, val_set, config, device, batches=8)
```

- `train` 在开始时才 `set_determinism(config.train.seed)`；它不控制此前
  `build_model` 的参数初始化。API 需要可复现时先自己调用 `set_determinism`。
- `TokenDataset` 读的是已经缓存的 token（先跑 `cache` 或 `cache_video_tokens`），
  划分来自 `config.data.train_videos/val_videos`，调用前要自己设置好。
- `train` 的 `out_dir` 必须是 `Path`。验证使用确定性后验均值，不消耗随机数。
- `train()` 只写出 checkpoint（`initial.pt`/`best.pt`/`final.pt`）、
  `train_log.jsonl` 与 `train_summary.json`；`config.json` 与 `splits.json` 是
  **CLI `train` 命令**写出的（见 [commands.md](commands.md)）。
- `--resume` / `resume=` 恢复优化器与 CPU/CUDA 随机状态；`max_steps` 是**总目标
  步数**，继续训练时要把它调到大于 checkpoint 的 step。
- `evaluate_model(model, val_set, ...)` 只读验证集缓存；`fit_baseline_stats(model,
  train_set, ...)` 额外需要训练集缓存来拟合基线统计量，`evaluate_baselines` 再用它
  在验证集上评分。CLI `eval` 两条路径都要，因此 train 与 val 缓存缺一不可。

## 单位

- `gaussian_nll_bits(target, mu, logvar)` 与 `gaussian_kl_bits(...)` 返回**逐元素**
  bits，形状 `(..., D)`（`logvar` 为对数方差）；由调用方决定在哪些维度上取均值。
  评价汇总为 **bits per dimension**（在 batch、patch、d_world 上取均值）。
- 时间一律为秒（float64）。实际落盘的量：`train_log.jsonl` 的
  `nll_bits_per_dim_horizon` / `nll_h<N>_bits_per_dim` / `mse` /
  `kl_bits_per_dim` / `kl_bits_per_event`；`metrics.json` 的
  `per_horizon`、`per_anchor_horizon`、`summary` 下的 `nll_bits_per_dim`、
  `mse`、`delta_seconds`；`train_summary.json` 的
  `best.val_mean_nll_bits_per_dim` 与 `final_val`。

## 来源与缓存

```python
from wpm_video import cache_video_tokens, load_source_manifest, resolve_source
from wpm_video.encoder import cache_path, cache_identity

meta = cache_video_tokens("clips/a.mp4", config.data, encoder, config.data.cache_dir)
record = resolve_source(info, load_source_manifest(config.data.video_dir))
```

`resolve_source` 返回一条记录：字段有 `kind`、`local_path`、`content_sha256`、
`source_id`。未在 `sources.json` 中声明的片段 `kind` 为 `local_file`，不会被归到
任何公开数据集；声明过的片段只有 SHA-256 与磁盘 bytes 一致时才是
`public_dataset`（另有 `dataset_repo_id/revision/path/url`），不一致时降级为
`local_file` 并记录 `demoted_from` 与 `demotion_reason`。

缓存路径由 `cache_identity(encoder)`（编码器身份 + 采样参数 + 时间轴 schema）决定，
写入与读取使用同一键。
