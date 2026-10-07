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
batch = encoder.encode_clips(clips)              # (B,T,3,H,W) uint8 -> [B × (P, d_model)]
tokens = tokens.to(device).unsqueeze(0)          # -> (1, P, d_model)，observe 需要这个形状
```

- `encode_clips` 是缓存路径使用的批处理入口：一个 clip 恰好是一个分块的采样帧，所以不会
  有分块看到后面分块的帧；`encode_clip` 等价于 `encode_clips(clip.unsqueeze(0))[0]`。
  只实现 `encode_clip` 的自定义编码器会自动退回逐条串行（`Encoder.encode_clips` 基类
  实现）。`cache_video_tokens` 的 `batch_clips` 就是真实 batch 大小。
- `vjepa2` 为冻结的真实编码器（推理在 `torch.no_grad` 下）；`native` 为随机初始化，
  需要 `allow_native=True`，只用于测试与离线示例。真实 run 不会静默退回 native。
  编码器始终以 FP32 运行：精度不是配置项，也不进缓存键与解码器兼容性记录。

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
  `delta_seconds == 0` 是 no-op：返回等值副本，此时不计算也不检查步数。
  `substeps=` 是给已经用 `substeps_for` 在主机侧算好步数的调用方的一致性检查：必须等于
  `ceil(max(delta)/substep_seconds)`，否则报错（校验从不跳过）。训练循环用的是内部
  plan 路径，只对 `gather_batch` 主机侧校验过的 batch 生效。
- `predict`：返回 `mu`、`logvar`（形状 `(B, P, d_world)`）、推进后的状态与注意力。
  `logvar` 是**对数方差**，范围由 `model.config.logvar_min/logvar_max` 限制；
  `sigma = exp(0.5 * logvar)`。`observe` 与 `predict` 都接受 batched 状态。
  `predict(..., want_attention=True)` 是默认值：即使 `performance.anchor_attention="sdpa"`
  也会走 reference 读以返回真实注意力权重；传 `False` 才使用配置的核并返回
  `attention=None`（训练/验证/推理内部路径都显式传 `False`）。
  `reference_mu=` 可复用同一状态已经算好的 present 估计（多 horizon 共享，梯度照常累加）。

## 流式与查询

```python
from wpm_video import stream_chunks, predict_at, predict_future, query_saved_state, demo

state, log = stream_chunks(model, encoder, frames, chunks[:3], device, state=None)
futures = predict_at(model, state, [1.0, 2.0])          # 只用状态与 Δ 秒
futures[2.0]["mu"], futures[2.0]["sigma"], futures[2.0]["target_time_seconds"]
result = demo(model, encoder, config, "clips/a.mp4", device, "runs/predict",
              prefix_chunks=3, state_path=None)          # 等价于 CLI predict
saved = query_saved_state(model, "state.pt", [1.0, 4.0], device, Path("runs/query"))

# 可选解码：demo 与 query_saved_state 接受同一个 decoder 与它的 payload
result = demo(model, encoder, config, "clips/a.mp4", device, "runs/predict_decoded",
              prefix_chunks=3, decoder=decoder, decoder_payload=payload)
result["decoded"]["predictions"]["records"]["h1"]        # 关键帧的元数据与 PNG 名
result["target_frames"][1]                               # 对应的真值帧（只用于对照图）
saved = query_saved_state(model, "state.pt", [1.0, 4.0], device, Path("runs/query_decoded"),
                          decoder=decoder, decoder_payload=payload, config=config)
```

解码是**可选**参数：不传 `decoder` 时 latent 产物与推理由此保持不变（`result["decoded"]`
为 `None`），`predict_summary.json` / `state_query.json` 只多出 `decoded_rgb` 这一段附加
元数据，未启用时为 `null`，既有键值不变。`query_saved_state` 在给了 `decoder` 却没给
`out_dir` 时抛 `ValueError`（解码要写文件，不能静默忽略）；给了 `decoder` 还必须给
`config`，因为兼容性检查需要编码器与采样身份。

查询 Δ 保持 float64：`predict_at` 的键、`delta_seconds` 与 `target_time_seconds` 都是调用
方给出的精确时间，`1.0` 与 `1.000000001` 是两个条目、两张不同文件名的图（只有激活值用
float32）。

- `stream_chunks` 与 `predict_at` 只处理 **batch 1** 的状态（内部取 `[0]`），并为每个
  分块产出一条日志：`skipped` 是被状态覆盖时的说明字符串（`"already covered by the
  state"`），正常观察时为 `False`。
- `predict_at` 返回以 Δ 秒为键的字典，每项含张量 `mu`、`sigma`（形状
  `(P, d_world)`），以及 `delta_seconds`、`target_time_seconds`、`scored`。
- `demo` 把张量写入 `future_latents.pt`（结构 `{"predictions": {horizon: {张量}},
  "meta": {...}}`，每个 horizon 只含 `mu`/`sigma`，有真值时含 `target`），把
  horizon、delta、时间戳与评分写入 `predict_summary.json`。
- `predict_at` / `query_saved_state` 不需要视频与编码器。

## 可选 RGB 解码器

```python
from wpm_video import (DecoderConfig, LatentRGBDecoder, build_decoder, decode_latents,
                       load_decoder, run_decoder_training, evaluate_decoder,
                       check_decoder_compatibility, ChunkFrameSource, alignment_record)

decoder = build_decoder(config.decoder, model.patches, model.config.d_world)
summary = run_decoder_training(config, "runs/run1/best.pt", Path("runs/decoder"), device)
decoder, payload = load_decoder("runs/decoder/best.pt")   # 默认加载到 CPU
decoder = decoder.to(device).eval()                       # 与 latent 同设备、推理模式

latents = {"h1": mu[0]}                        # {"键": (P, d_world)}
images = decode_latents(decoder, latents, device)      # {"h1": (3, S, S) float，[0, 1]}
payload["path"]                                # 该 checkpoint 的绝对路径（加载时写入）
```

`decode_latents(decoder, latents, device)` 会把**解码器与 latent 都**移到 `device` 再推理
（因此 CPU 上加载的解码器也能解码 GPU 上的 latent），并恢复调用前的 train/eval 模式；
解码器之后留在该设备上，不会悄悄被移回。上面的 `.to(device)` 不是必需但更直观。

解码器是**可替换组件**：`config.decoder.kind` 选择架构，`"conv"` 是内置默认值。

```python
from wpm_video import RGBDecoder, available_decoders, register_decoder

class MyDecoder(RGBDecoder):        # 契约：(B, P, d_world) -> (B, 3, S, S) in [0, 1]
    def __init__(self, config, patches, d_world):
        super().__init__(config, patches, d_world)     # 校验并保存共享几何
        ...
    def forward(self, latents):
        self.check_input(latents)
        ...

register_decoder("myarch", MyDecoder)     # 进程内有效；未知 kind 一律报错
assert "myarch" in available_decoders()
```

| 调用 | 返回 | 说明 |
|---|---|---|
| `register_decoder(kind, factory)` | `factory` | 注册 `factory(config, patches, d_world) -> RGBDecoder`；可当装饰器用。名字只注册一次（重复、含覆盖 `conv` 都报错），进程内有效，不做插件发现。kind 标识**实现及其语义**：语义变了要换新名字（`my_decoder_v2`）并保留旧名以读取旧权重 |
| `available_decoders()` | `tuple[str, ...]` | 当前进程已注册的 kind（有序） |
| `decoder.kind` / `.config` / `.patches` / `.d_world` / `.grid` / `.output_size` | `str` / `DecoderConfig` / `int` / `int` / `(gh, gw)` / `int` | 组件元数据；`config.options` 与整个 config 都是构建时的深拷贝快照 |
| `decoder.check_input(latents)` | `None` | 输入契约；架构可在 `forward` 里先调用它 |
| `decoder.parameter_count()` | `int` | 参数量，与 checkpoint 记录的一致 |

契约、注册生命周期、checkpoint schema 与对照实验协议见
[decoder_components.md](decoder_components.md)。

| 调用 | 形状 / 类型 | 说明 |
|---|---|---|
| `build_decoder(config.decoder, patches, d_world)` | `RGBDecoder`（`kind="conv"` 时是 `LatentRGBDecoder`） | 按 `config.decoder.kind` 分发到已注册工厂；构建时即校验配置、网格与工厂返回的元数据；非法配置或未注册 kind 直接抛 `ValueError`（`DecoderRegistrationError` 是其子类） |
| `decoder(latents)` | `(B, P, d_world)` -> `(B, 3, S, S)` | 输入形状不对（`P`/`d_world`/维度）抛 `ValueError`；输出是 `[0, 1]` |
| `decoder.parameter_count()` / `decoder.output_size` / `decoder.grid` | `int` / `int` / `(gh, gw)` | 参数量、输出边长、latent 网格 |
| `decode_latents(decoder, {键: (P, d_world)}, device)` | `{键: (3, S, S)}` float `[0, 1]` | 把解码器与 latent 都移到 `device`，推理模式下运行并恢复原 mode；键可以是 horizon 或 Δ 秒 |
| `render_latents(decoder, {键: (latent, meta)}, out_dir, device, payload=..., prefix=...)` | dict | 写 `<prefix>_<键>.png` 与 `frames` 张量；两个键映射到同一文件名时报错 |
| `to_uint8(image)` / `save_frame_png(image, path)` | `(S, S, 3)` uint8 / 路径 | `[0,1]` float 三通道 RGB -> PNG（磁盘上是 BGR 字节序） |
| `run_decoder_training(config, world_checkpoint, out_dir, device, resume="")` | dict | 见下 |
| `evaluate_decoder(world_model, decoder, dataset, frame_source, config, device, batches, reference=None)` | dict | `l1`/`mse`/`psnr_db`/`per_video`；只读验证集，消耗零随机数 |
| `load_decoder(path)` | `(RGBDecoder, payload)` | 默认在 CPU 上构建，按 checkpoint 记录的 `kind` 分发；`payload["path"]` 是绝对路径；非解码器 checkpoint、不支持的 schema、未注册 kind、或元数据/权重不自洽时抛 `ValueError`（**在读入权重之前**） |
| `check_decoder_compatibility(payload, model, config, decoder=None)` | dict | 不匹配抛 `DecoderCompatibilityError`（`ValueError` 子类） |

`run_decoder_training` 的输入是**世界模型 checkpoint 路径**，内部按顺序：加载并冻结世界
模型 → 校验调用方 config 与 checkpoint 记录的编码器/采样身份一致 → （续训时）以
checkpoint 架构为准 → 从 provenance 取划分并校验缓存、源视频与时间轴 → 训练。产物见
[commands.md](commands.md#train-decoder)。`--resume` 恢复优化器、步数、两个采样生成器与
CPU/CUDA 随机状态，因此续训与不中断的训练逐位一致。

单位：像素指标是 `[0, 1]` 上的均值，`psnr_db = 10*log10(1/mse)`；时间一律为秒。解码产物
中每张图都是**一个分块最后一帧**的关键帧，按 horizon/Δ 排列是阅读顺序，不是连续视频。
细节见 [decoder.md](decoder.md)。

## 目标帧与对齐

```python
from wpm_video import ChunkFrameSource, alignment_record
source = ChunkFrameSource(config.data.video_dir, config.data, config.decoder.image_size,
                          max_videos=config.decoder_train.frame_cache_videos)
source.register(name, alignment_record(cache, require_identity=True))   # 训练路径
frame, timestamp = source.target_frame(name, cache.chunks[index])       # (3,S,S) uint8, 秒
```

- 按视频惰性解码，最多同时在内存里保留 `max_videos` 条（LRU 淘汰）；训练路径必须先
  `register`，`require_identity=True` 会拒绝缺少内容哈希/分块记录的缓存；
- `target_frame` 用缓存 payload 里的 `end_frame - 1`（即该分块最后被采样的一帧），并断言
  它的时间戳等于 `end_seconds - 1/source_fps`；源视频字节、采样栅格或分块边界与缓存不一致
  时抛 `CacheAlignmentError`。

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

## 性能选项

```python
from wpm_video import (PerformanceConfig, apply_compile, autocast_context, build_optimizer,
                       materialize_stats, should_pin, substeps_for, to_device)

config.performance = PerformanceConfig(precision="bfloat16", anchor_attention="sdpa",
                                       fused_optimizer=True, compile=False, pin_memory=True)
```

| 调用 | 说明 |
|---|---|
| `PerformanceConfig(...)` | 加速开关；默认全部是参考实现。字段与边界见 [performance.md](performance.md) |
| `autocast_context(performance, device)` | `bfloat16` 时返回 autocast 上下文，否则 no-op；`device` 可传字符串或 `None` |
| `build_optimizer(params, performance, lr, wd, device)` | AdamW；`fused_optimizer` 在非 CUDA 上抛 `PerformanceError`，不会静默退回 |
| `apply_compile(model, performance, compile_fn=None)` | 编译 `compile_scope` 指定的普通可调用对象，不改 `state_dict`；`training_blocks` 做前向/反向预检并原子挂接。返回实际 scope、targets 和状态；显式允许设置阶段回退时为 `fallback`，否则失败抛 `PerformanceError` |
| `to_device(tensor, device, performance=None)` | 按策略搬运（CUDA 上可选 pinned/非阻塞）；`performance=None` 即参考行为 |
| `substeps_for(deltas, substep_seconds, max_substeps)` | 主机侧按与积分器相同的公式算步数 |
| `materialize_stats(stats)` | 把 `forward_window(..., stats_mode="tensor")` 的统计量转成 float |

数值契约：bfloat16 只作用于模型计算；持久状态 FP32、时钟 float64、NLL/KL 与投影标准化
以及所有上报指标都在 FP32；冻结编码器 FP32。检查点记录实际生效的策略，续训拒绝语义项
（精度/注意力核/fused/compile/生效编译范围）变化，忽略设备与 pinned 等元数据差异。

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
