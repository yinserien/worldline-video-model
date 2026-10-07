# 解码器组件：接入自定义架构

RGB 解码器是**可替换组件**。默认且唯一内置的架构仍是原来的卷积上采样器
（`kind="conv"`）：网络、参数名、参数量与初始化顺序不变，v0.4.0 训练出的权重照常加载，
旧的配置与导入路径也保持可用（写入格式升级为 schema 2，见
[checkpoint 格式与兼容](#checkpoint-格式与兼容)）。本文说明如何接入**第二种架构**，并让
它在同一条训练 / 评价 / 保存 / 加载 / 渲染链路上工作。

先明确边界，避免误解：

- 组件只负责 `(B, P, d_world)` latent → `(B, 3, S, S)` 的一帧 RGB，**不参与**动力学、
  状态、投影与编码器：世界模型、持久状态与动力学都不 import 解码器，解码只发生在推理
  末尾、作用于已经算好的 latent（`predict` 只是在最后一步调用渲染助手把 latent 写成图）。
- 换架构**不是**换质量承诺：默认 conv 仍是默认；"另一种架构更锐利"是一个**尚未测量**的
  假设，只有在固定数据/划分/投影/输出分辨率下测出来才算数（见
  [对照实验协议](#对照实验协议)）。
- 不引入新依赖，不做 GAN / diffusion / 感知损失 / 自定义算子。

## 目录与职责

```
src/wpm_video/decoder/
    base.py              RGBDecoder 基类：公开契约与共享几何校验（不含任何架构假设）
    registry.py          register_decoder / available_decoders / build_decoder 按 kind 分发
    architectures/conv.py 内置 conv 架构（kind="conv"），导入即注册
    model.py             checkpoint 格式（schema 1/2）、save_decoder / load_decoder、
                         以及旧的 LatentRGBDecoder / build_decoder 导入路径
    compat.py            投影指纹、编码器/采样身份、架构与 kind 比对
    targets.py           目标帧读取与时间轴对齐
    train.py             独立的解码器训练/评价/产物
    render.py            latent -> PNG 与张量产出的共享渲染助手
```

依赖方向是单向的：`config` 不 import 解码器，解码器组件模块（`base`、`registry`、
`architectures/*`）不 import 动力学或训练代码。`registry` 也**不** import
`architectures`——内置 conv 由包初始化时显式导入注册，避免循环依赖。

## 组件契约

```python
from wpm_video import DecoderConfig, RGBDecoder, register_decoder

class MyDecoder(RGBDecoder):
    def __init__(self, config: DecoderConfig, patches: int, d_world: int):
        super().__init__(config, patches, d_world)   # 校验并保存共享几何
        ...                                          # 只创建自己的参数

    def forward(self, latents):                      # (B, P, d_world) -> (B, 3, S, S)
        self.check_input(latents)
        ...

register_decoder("myarch", MyDecoder)
```

`RGBDecoder` 提供并保证：

| 成员 | 含义 |
|---|---|
| `config` | 构建它的 `DecoderConfig`（`options` 是**快照**，调用方之后改原字典不影响它） |
| `patches` / `d_world` / `grid` | patch 数、latent 宽度与 `(gh, gw)` 网格，来自世界模型 |
| `output_size` | 输出边长 `S`，等于 `config.image_size` |
| `parameter_count()` | 参数量（checkpoint 与摘要记录的就是它） |
| `kind` | 构建它的 kind，等于 `config.kind` |
| `check_input(latents)` | 输入契约：3 维、浮点、`P`/`d_world` 与 checkpoint 一致、batch ≥ 1 |

基类校验**共享几何**：patch 网格是正方形、`d_world ≥ 1`、`image_size` 是合法的公共输出
边长（`[32, 2048]` 且为 8 的倍数）。**怎么产生输出是架构自己的事**：conv 要求
`image_size / 网格边长` 是 ≥ 1 的 2 的幂（每级上采样必须精确），而一个先在小分辨率上生成
再插值/重排的架构可以接受任何合法输出边长（包括比网格还小）。基类不知道 stem、卷积、
通道数或上采样级数。

输出必须是 `(B, 3, S, S)`、取值 `[0, 1]`，并且只依赖输入 latent：不看世界状态、不看
时间戳、不读文件。模块 `train()`/`eval()` 行为**可以**不同（BatchNorm、dropout 都是合法
的）：渲染与评价会在 `eval()` 下运行（`decode_latents` 会切换并恢复原模式），训练在
`train()` 下运行；内置 conv 没有这类层，两种模式下像素完全相同。

共享训练和评价检查输出形状与浮点类型，避免错误的通道数被广播成 RGB。渲染在 CPU 上
进一步检查数值有限且处于 `[0, 1]`；发生错误时也会恢复原有训练模式。

工厂返回的对象会被检查：kind、`patches`/`d_world`/`grid`、`output_size`、以及它报告的
`config` 必须与请求一致，否则 `build_decoder` 直接报错（而不是写出一个以后加载不回来的
checkpoint）。

## 注册的生命周期

```python
from wpm_video import available_decoders, register_decoder
register_decoder("myarch", MyDecoder)   # 只在本进程内有效
available_decoders()                    # ('conv', 'myarch')
```

- **进程内注册**：注册不是持久的，也不做插件发现。任何要构建、加载或训练该架构的进程
  都必须先 import 定义它的模块（或显式调用 `register_decoder`）。
- **不按 checkpoint/config 里的字符串动态 import**，也不下载任何东西：未知 kind 一律
  报错，并列出当前已注册的 kind。
- **名字只注册一次**：同一进程内重复注册（包括覆盖内置 `conv`）会被拒绝。kind 语法为
  小写标识符（`[a-z][a-z0-9_]*`，≤ 64 字符）。
- **kind 标识的是"实现（含语义）"，不只是张量形状或版本号**，因此跨进程/跨机器的保证来自
  作者，而不是注册表：详见下一节。注册表只能保证"同一进程内这个名字只被绑定一次"。
- **stock CLI 默认只用内置 conv**。要让 `wpm-video train-decoder` / `predict` / `query`
  使用自定义架构，就在**自己的 Python 入口**里先注册再调用 CLI：

```python
from my_project.my_decoder import MyDecoder          # 导入即注册（或显式 register_decoder）
from wpm_video.cli import main
raise SystemExit(main(["train-decoder", "--config", "config.json",
                       "--checkpoint", "runs/run1/best.pt", "--out", "runs/decoder"]))
```

## kind 的语义与版本

`kind` 是**实现的名字，而不是形状的标签**：它承诺"这个名字下的网络及其数学含义"。于是有
两条作者侧规则：

- 只要改了"这个 kind 算什么"（即使参数形状完全不变，例如换了激活、改了归一化位置、改变了
  通道重排方式），就**必须**发一个新名字（`my_decoder_v1` → `my_decoder_v2`），并**同时
  保留** v1 的注册，让旧 checkpoint 仍能按原语义读出；
- 只是同一个实现的普通 bugfix/性能改动，且语义与权重一一对应时，才可以沿用原名。

原因是加载侧不做任何猜测：checkpoint 里的 `kind` 只会交给**本进程已注册**的同名工厂，
既不动态 import 代码，也不按形状反推架构。注册表能保证"同一进程内一个名字只绑定一次"，
跨机器的一致性只能由作者用命名版本维护。

示例里的 `pixelshuffle_v1` 就是按这条规则命名的。

## 配置选择

```json
{
  "decoder": {
    "kind": "my_decoder_v1",
    "image_size": 128,
    "options": {"hidden": 64}
  }
}
```

| 字段 | 说明 |
|---|---|
| `kind` | 已注册的组件名；默认 `"conv"`。**只做语法校验**（不需要注册表），未知 kind 在 `build_decoder`/`load_decoder` 时报错 |
| `image_size` | 所有架构共享的输出边长：`[32, 2048]` 且为 8 的倍数；更严的约束（如 conv 的整数倍/2 的幂）由架构自己检查 |
| `options` | 该架构自己的设置：JSON 兼容、键为字符串、数值有限；**必须**是 dict（缺省 `{}`）。名字空间是隔离的，`kind`/`image_size`/`grid` 之类的名字可以随便用，见下 |
| `base_channels` / `channel_multipliers` / `stem_blocks` / `blocks_per_stage` | **仅供 conv**。`kind != "conv"` 时必须保持默认值，否则报错（避免记录一个实际不生效的设置） |

`kind` 与 `options` 追加在原有字段**之后**，所以位置参数写法
`DecoderConfig(128, 128, [1, 2, 2], 2, 1)` 含义不变；旧 `config.json`（没有这两个键）
加载后得到 `kind="conv"`、`options={}`。

`options` 必须能**原样通过 JSON 往返**：容器只能是 dict 与 list（tuple、set、tensor、任意
对象、非 dict 的 Mapping 一律拒绝），键是字符串，数字有限；循环引用会作为错误报出，而不是
递归崩溃。共享层只做这件事，语义由架构自己解释与校验（例如示例只接受
`{"hidden": int}`，拼错的键会报错而不是被忽略）。

**架构身份**记录为嵌套结构：

```json
{"kind": "my_decoder_v1", "image_size": 128, "options": {"hidden": 64}}
```

conv 仍然是 `{"kind": "conv", "image_size": ..., "base_channels": ..., ...}`（它的
`options` 强制为空，所以没有嵌套块）。`options` 单独成块的好处是名字空间天然隔离：某个
架构有个叫 `grid` 或 `kind` 的选项也不会与身份字段冲突，顶层 `kind`/`image_size` 永远来自
配置本身，不会被选项覆盖。

## 训练、保存、加载、渲染

自定义组件走的是同一条路径，没有任何 conv 专用分支：

```python
from pathlib import Path
from wpm_video import load_decoder, run_decoder_training, decode_latents

summary = run_decoder_training(config, "runs/run1/best.pt", Path("runs/decoder"), device)
decoder, payload = load_decoder("runs/decoder/best.pt")     # 按 kind 分发到已注册工厂
images = decode_latents(decoder, {1: mu}, device)           # {1: (3, S, S) in [0, 1]}
```

- 训练前照旧校验：调用方 config 与 world checkpoint 的编码器/采样身份一致、缓存与源视频
  对齐、划分来自 world provenance。
- `train_summary.json` 的 `decoder` 段报告**组件**而不是 conv 字段：
  `{"kind", "architecture", "parameters", "image_size", "grid", "d_world"}`。
- `--resume` / `train_decoder(resume=...)` 比对的是**规范化架构记录（含 kind 与 options）**，
  不一致会在第一个梯度步之前报错；checkpoint 的架构是权威，config 只能与它一致。
- 渲染（`render_latents` / `predict` / `query`）只调用 `decoder(latents)`，与架构无关；
  产物里的 `decoder.component_kind` 会写明是哪个组件生成的。

## checkpoint 格式与兼容

| schema | 何时写入 | 内容 |
|---|---|---|
| **1** | v0.4.0 及更早 | 没有 `kind`/`options`：那时只可能有一个架构，即 conv |
| **2** | 现在 | `decoder_config` 显式记录全部字段（含 `kind`/`image_size`/`options`，conv 另有四个形状字段），`architecture` 记录组件身份 |

读取规则（两个版本都严格校验）：

- schema 1 被**明确**解释为"旧 conv"：它必须带齐全部四个 conv 形状字段，且**不得**出现
  `kind`/`options`；缺字段或多字段一律报错，不会用默认值补齐、也不会被"修好"成某个架构。
- schema 2 的 `decoder_config` 必须**显式**包含共享字段（`kind`/`image_size`/`options`；
  conv 还必须包含四个形状字段）：截断的记录是错误，不会被默认值悄悄补全成另一个网络。
  `schema_version` 必须是整数（`true`/`"2"` 都不算）。
- `architecture` 记录必须与 `decoder_config` 一致（`patches`/`d_world`/`grid` 也要互相
  自洽），权重必须能装进按记录构建的模块。
- 未知/未来 schema、未知 kind（本进程未注册）、被改写过的元数据：全部在**产生任何像素或
  梯度之前**报错。
- 兼容性检查（`check_decoder_compatibility` / `identity_gaps`）对两个版本都生效，并继续
  强制投影指纹、编码器身份、采样参数与目标帧语义/分辨率；此外还会比对组件 kind 与规范化
  架构，**维度相同但架构不同不再可能被接受**。即使没有传解码器模块，检查也会验证
  checkpoint 自身记录的配置与架构彼此一致。
- 附加元数据（provenance、优化器状态、运行 config）写在格式字段旁边，**不能覆盖**
  `schema_version`/`decoder_config`/`architecture`/`patches`/`d_world`/`grid`/`state_dict`
  等格式字段。

旧的 `LatentRGBDecoder`、`build_decoder`、`save_decoder`、`load_decoder`、
`group_count`、`DECODER_SCHEMA_VERSION` 导入路径全部保留；`LatentRGBDecoder` 仍是同一个
conv 网络（类体只是搬到了 `architectures/conv.py`），参数名、形状、参数量与初始化顺序
不变，v0.4.0 训练出的解码器权重照常加载。

## 对照实验协议

组件 API 只保证"可替换、可比较"，**不保证**替换后更好。要得到可信结论：

1. **固定**数据与划分（同一个 world checkpoint 的 provenance 与同一份 token 缓存）、
   同一个投影、同一个 `image_size`、同一套训练超参与步数——只改 `kind`/`options`；
2. 报告**留出视频**上的 `l1`/`mse`/`psnr_db`、`per_video` 与 `constant_frame_reference`
   （常量帧参考由**训练集**拟合）；
3. 把**真值 latent 的重建**（`target_reconstruction_*`，衡量解码器本身）与**预测 `mu` 的
   解码**（`decoded_*`，等于预测误差 + 解码误差）分开报告，不要混为一谈；
4. 多次运行同一配置（不同 seed）再比较，单次结果不作结论；
5. 结论必须同时说明解码器只输出**每分块最后一帧**的关键帧，且重构自冻结编码器 + 随机投影，
   本身有损。

`examples/custom_decoder.py --compare-conv` 会把上面第 1–3 条跑一遍（合成数据、CPU、
离线），打印两个架构的同一套指标；那只是协议演示，不是质量结论。注意 conv 只接受
"网格边长 × 2 的幂"的输出边长，所以对照必须显式给一个两边都合法的尺寸（4x4 网格用
`--image-size 64`）；否则脚本会明确说明**没有做任何对照**并以非零状态退出，而不是拿一次
被跳过的运行冒充并排结果。

## 可运行示例

`examples/custom_decoder.py` 定义了一个**没有任何卷积**的第二架构（逐 cell MLP +
pixel shuffle，`kind="pixelshuffle_v1"`），并在导入时注册：

```bash
# --out 必须是包外的工作目录（脚本会拒绝仓库/包目录内的路径）
# 完全离线：合成三段小视频、训一个 tiny native 世界模型，再训自定义解码器并渲染一帧
python examples/custom_decoder.py --out /tmp/wpm_run/custom_demo

# 用已有 world checkpoint（仍然离线，用本地视频）
python examples/custom_decoder.py --out /tmp/wpm_run/custom_demo \
    --config /tmp/wpm_run/config.json --checkpoint /tmp/wpm_run/runs/world/best.pt

# 在本进程注册组件后，直接驱动 stock CLI（--device 同样会生效）
python examples/custom_decoder.py --out /tmp/wpm_run/custom_demo --via-cli

# 同数据/同划分/同投影/同输出分辨率地跑一遍内置 conv，并排打印指标
python examples/custom_decoder.py --out /tmp/wpm_run/custom_demo \
    --image-size 64 --compare-conv
```

`--out` 是必填项，也是该脚本唯一写入的位置（离线夹具放在系统临时目录）；仓库检出目录与
安装目录内的路径一律被拒绝。示例默认 `--image-size 48`：4x4 网格下 `48/4 = 12` 不是 2 的
幂，conv 会拒绝而该架构接受（它只要求整数倍）——这正是"输出几何约束属于架构、不属于契约"
的直接演示。

## 不做的事（当前范围）

- 不引入 GAN / diffusion / 感知或对抗损失，不新增依赖，不写自定义算子/内核；
- 不改变动力学、投影、编码器、目标帧语义或既有 latent 产物；
- 不做自动插件发现、不做按 checkpoint 字符串的 import、不做隐式下载；
- **默认仍是 conv**；另一种架构是否更锐利，要按上面的协议测量后再谈。
