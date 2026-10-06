# 公开样例片段：下载与来源校验

包本身可以使用任意本地 `.mp4`。下面的脚本从固定 revision 下载一小段公开样例片段
并生成**逐字节校验**过的来源清单；如果你只用自己的视频，可以跳过本页。

请在**包外**准备环境与工作目录：包目录保持只读，不要在其中创建 `.venv`、`cache`
或 `runs`。

```powershell
python -m venv --system-site-packages E:\work\wpm_env
& "E:\work\wpm_env\Scripts\python.exe" -m pip install "E:\path\to\worldline_video_model-0.3.0-py3-none-any.whl[vjepa]"
New-Item -ItemType Directory -Force E:\work\wpm_run | Out-Null
Set-Location E:\work\wpm_run
$py = "E:\work\wpm_env\Scripts\python.exe"
```

Linux / macOS 把解释器换成 `~/wpm_env/bin/python`、路径改为正斜杠。

## 1. 下载并校验

脚本随源码发行包提供（wheel 内不含 `examples/`），用完整脚本路径运行：

```powershell
& $py "E:\path\to\sdist\examples\prepare_samples.py" --out videos --clips 20
```

脚本行为：

- 固定 `nateraw/kinetics-mini` 的 revision
  `9f4ed38128a355c352527209101be3e326471816`，只取 `train/archery` 与
  `train/bowling` 两个目录（不会误取缓存里的其他类别）；
- 已下载过的快照直接复用本地缓存，不会重复下载；
- 复制到 `videos/` 时保留原始文件名（`<类别>_<源文件stem>.mp4`），源 identity 不丢失；
- 每个文件与快照原文件做 SHA-256 比对，只有一致的片段才写进 `videos/sources.json`，
  其余列入 `skipped`。

只复核已有集合、不复制：

```powershell
& $py "E:\path\to\sdist\examples\prepare_samples.py" --out videos --check-only
```

## 2. 接着训练与推理

编码器权重在第一次 `cache` 或 `predict` 时自动联网下载并进入 Hugging Face 缓存，
之后复用；不需要预先手动下载。生成配置（默认 `data.video_dir` 就是 `videos`）：

```powershell
& $py -c "from wpm_video import RunConfig; RunConfig().save('config.json')"
& $py -m wpm_video cache --config config.json
& $py -m wpm_video train --config config.json --out runs\run1 --val-count 4
& $py -m wpm_video eval --config config.json --checkpoint runs\run1\best.pt --out runs\eval

# 从配置与划分文件里取真正的验证视频路径
$config = Get-Content config.json -Raw | ConvertFrom-Json
$splits = Get-Content runs\run1\splits.json -Raw | ConvertFrom-Json
$videoPath = Join-Path $config.data.video_dir ($splits.val[0] + ".mp4")
& $py -m wpm_video predict --config config.json --checkpoint runs\run1\best.pt `
    --video $videoPath --out runs\predict --prefix-chunks 3
& $py -m wpm_video query --config config.json --checkpoint runs\run1\best.pt `
    --state runs\predict\world_state.pt --deltas 1 2 4 8 --out runs\query
```

Linux / macOS 把 `& $py -m wpm_video` 换成 `~/wpm_env/bin/python -m wpm_video`、
路径改为正斜杠，验证视频路径用
`$(python -c "import json;c=json.load(open('config.json'));s=json.load(open('runs/run1/splits.json'));print(c['data']['video_dir']+'/'+s['val'][0]+'.mp4')")`
构造。

## 3. 用你自己的片段

不用本页脚本，直接把 `.mp4` 放进 `videos/`（或配置里的 `data.video_dir`）即可，
这些片段会记为 `local_file`。只有确实来自公开数据集时才用 `wpm-video provenance`
声明，并先通过上面的字节校验。若不需要真实编码器，可以改用
`configs/native_tiny.json`（CPU、无下载、输入为脚本生成的合成视频），直接运行：

```powershell
& $py "E:\path\to\sdist\examples\native_end_to_end.py" --out E:\work\wpm_run\native_demo --steps 40
```
