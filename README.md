# worldline-video-model

提供视频编码缓存、训练、流式预测与状态查询：按因果分块读取视频，在块之间保持一个
持久状态，并按真实时间预测未来时刻的表征分布。预测输出为 latent 的均值与标准差，
不生成 RGB。包内不包含权重与视频。

## 安装

需要 Python ≥ 3.10，且 `torch` 可用。包目录本身保持只读即可：请在**包外的目录**
建虚拟环境与工作目录，不要在包内创建 `.venv`、`cache` 或 `runs`。

```powershell
# 1) 包外环境（可复用已有的 torch，避免重复下载）
python -m venv --system-site-packages E:\work\wpm_env
# 2) 安装发行包
& "E:\work\wpm_env\Scripts\python.exe" -m pip install "E:\path\to\worldline_video_model-0.1.0-py3-none-any.whl"
# 3) 校验（用解释器全路径，不依赖 PATH）
& "E:\work\wpm_env\Scripts\python.exe" -m wpm_video --version
```

Linux / macOS：

```bash
python -m venv --system-site-packages ~/wpm_env
~/wpm_env/bin/python -m pip install "/path/to/worldline_video_model-0.1.0-py3-none-any.whl"
~/wpm_env/bin/python -m wpm_video --version
```

- 安装 wheel 后源码树不受影响。也可以从源码目录 `python -m pip install .`
  （构建过程中源码目录里会出现构建中间产物）。
- 只用 native 编码器（合成视频、CPU）时，上面的安装就够了。
- 需要真实 V-JEPA 2 编码器时，再安装可选依赖：安装 wheel 时写成
  `"E:\path\to\worldline_video_model-0.1.0-py3-none-any.whl[vjepa]"`，或单独执行
  `pip install "transformers>=5.15,<6" huggingface_hub`。
- 本页后续命令统一写作 `& $py -m wpm_video`，其中 `$py` 是解释器全路径
  （`&` 是 PowerShell 调用运算符，省略它无法执行变量里的命令）。激活环境后也可以用
  `wpm-video` / `python -m wpm_video`，但控制台脚本是否在 PATH 上取决于环境，
  用解释器全路径最稳。

首次使用真实编码器 cache 或 predict 时会**自动联网下载**权重（约 1.2 GB，缓存到
Hugging Face 缓存目录），之后复用本地缓存，无需重复下载。想提前下载可以自己执行
（可选，不依赖 `hf` 是否在 PATH 上）：

```powershell
& $py -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/vjepa2-vitl-fpc64-256', revision='b3c1679b7c34d3255ef3547f27c7b226aefab26f')"
```

## 工作目录与配置

所有相对路径都相对**运行时 cwd** 解析，与包安装位置无关。配置可以从任意 cwd 生成：

```powershell
New-Item -ItemType Directory -Force E:\work\wpm_run | Out-Null
Set-Location E:\work\wpm_run
$py = "E:\work\wpm_env\Scripts\python.exe"
& $py -c "from wpm_video import RunConfig; RunConfig().save('config.json')"
```

生成的 `config.json` 使用默认值（`data.video_dir` = `videos`，`cache_dir` =
`cache/tokens`，设备 `cuda`）。把它改成你的实际输入位置和采样参数后再往下走；字段
说明、默认值与约束见 [docs/configuration.md](docs/configuration.md)。

## 无下载的 CPU 示例

示例脚本随源码发行包提供（wheel 内不含 `examples/`）。用随机初始化的 native 编码器
和脚本现场生成的合成视频跑完整链路（cache → train → 新进程 resume → eval →
predict → query），不需要 GPU 和任何下载：

```powershell
Set-Location E:\work\wpm_run
$py = "E:\work\wpm_env\Scripts\python.exe"
& $py "E:\path\to\sdist\examples\native_end_to_end.py" --out E:\work\wpm_run\native_demo --steps 40
```

输入是合成视频、编码器是随机权重，该示例只用来确认安装与链路可用。

## 用自己的视频

把 `.mp4` 放进工作目录的 `videos\`（或改 `data.video_dir`）。以下步骤从上面生成的
`config.json` 出发，可直接复制执行。

```powershell
Set-Location E:\work\wpm_run
$py = "E:\work\wpm_env\Scripts\python.exe"

# 1) 编码并缓存分块（编码器只读取每个分块内的帧；首次会下载冻结编码器权重）
& $py -m wpm_video cache --config config.json

# 2) 训练：按整段视频留出 1 条作验证集
& $py -m wpm_video train --config config.json --out runs\run1 --val-count 1

# 3) 评价：默认评估同一 run 的 initial/best/final，并与三条基线对比
& $py -m wpm_video eval --config config.json --checkpoint runs\run1\best.pt --out runs\eval

# 4) 从划分文件与配置里取出真正的验证视频路径，不要手工拼
$config = Get-Content config.json -Raw | ConvertFrom-Json
$splits = Get-Content runs\run1\splits.json -Raw | ConvertFrom-Json
$videoPath = Join-Path $config.data.video_dir ($splits.val[0] + ".mp4")

# 5) 流式预测：只观察前缀分块，之后按真实时间预测未来
& $py -m wpm_video predict --config config.json --checkpoint runs\run1\best.pt `
    --video $videoPath --out runs\predict --prefix-chunks 3

# 6) 状态查询：不需要视频与编码器
& $py -m wpm_video query --config config.json --checkpoint runs\run1\best.pt `
    --state runs\predict\world_state.pt --deltas 1 2 4 8 --out runs\query
```

继续训练时**沿用上一个 run 的 `config.json`**（它记录了真实的 train/val 划分），
并把 `max_steps` 调到大于 checkpoint 的 step，否则不会继续训练：

```powershell
$py = "E:\work\wpm_env\Scripts\python.exe"
& $py -c "from wpm_video import RunConfig; c = RunConfig.load('runs/run1/config.json'); c.train.max_steps = 6000; c.save('runs/run2_config.json')"
& $py -m wpm_video train --config runs\run2_config.json --out runs\run2 --resume runs\run1\final.pt
```

划分优先级：命令行同时给出 `--train-videos`/`--val-videos` > 配置文件里的
`data.train_videos`/`data.val_videos` > 按文件名顺序自动留出 `--val-count` 条。
训练与验证不得共享同一文件或同一来源 identity。`initial.pt` 只在新训练时写出，
`best.pt` 只在验证指标改善时写出。

`predict` 只需要 checkpoint、编码器配置和视频，可以搬到没有训练数据的机器上运行；
`query` 连视频和编码器都不需要。各命令参数与产物见 [docs/commands.md](docs/commands.md)。

## Python API

```python
import torch
from pathlib import Path
from wpm_video import (RunConfig, VideoWorldModel, build_chunks, build_encoder, predict_at,
                       probe_video, read_frames, stream_chunks)

config = RunConfig.load("config.json")        # 接受 str 或 Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, payload = VideoWorldModel.from_checkpoint("runs/run1/best.pt")
model = model.to(device).eval()               # 推理：后验取均值，确定性

video = "videos/a.mp4"
info = probe_video(video)
frames, timestamps = read_frames(video, config.data.fps, config.data.image_size)
chunks = build_chunks(info, config.data, timestamps)

encoder = build_encoder(config.encoder, device)          # 冻结的 V-JEPA 2
state = model.initial_state(1, device)
state, log = stream_chunks(model, encoder, frames, chunks[:3], device, state=state)
state.save("state.pt")                                   # 状态可跨进程保存

futures = predict_at(model, state, [1.0, 2.0, 4.0])      # 只用 state 与时间查询
mean, sigma = futures[2.0]["mu"], futures[2.0]["sigma"]  # 形状 (patches, d_world)
```

其中 `stream_chunks`、`predict_at` 只处理 batch 1 的状态；底层的
`model.observe` / `model.predict` 支持 batched 状态。完整 API、张量形状与路径参数
类型见 [docs/api.md](docs/api.md)。

## 使用文档

- [docs/configuration.md](docs/configuration.md)：配置字段、默认值、约束与路径规则
- [docs/commands.md](docs/commands.md)：CLI 命令、参数与输出文件
- [docs/api.md](docs/api.md)：Python API 与张量形状、单位
- [examples/prepare_sample_dataset.md](examples/prepare_sample_dataset.md)：公开样例片段下载与来源校验

## 目录

```
src/wpm_video/   包源码（config data dataset encoder model world_state train evaluate
                 predict viz selfcheck cli __main__）
configs/         vjepa2_256.json（真实编码器）、native_tiny.json（CPU，无下载）
examples/        无下载全链路示例、公开样例数据准备
tests/           架构契约、端到端流水线、CLI 与来源校验
docs/            使用文档
```
