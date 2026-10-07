"""Run configuration: data, encoder, model, training and optional decoder settings.

Plain dataclasses with JSON round-tripping plus strict validation. Every field
here is used by the implementation, and ``validate`` refuses combinations that
would silently change the meaning of a run (bad dimensions, illegal horizons,
overlapping splits, unstable integration).

The ``decoder`` and ``decoder_train`` sections configure the *optional* RGB
decoder. They are absent from every configuration written before the decoder
existed and default to valid values, so an old ``config.json`` loads unchanged
and the world-model commands behave exactly as before.

``decoder.kind`` and ``decoder.options`` select the decoder *component*
(``"conv"``, the built-in convolutional upsampler, is the default). They are
appended after the original fields, so a configuration written earlier keeps its
exact meaning; see docs/decoder_components.md.
"""

from dataclasses import MISSING, asdict, dataclass, field
import json
import math
from pathlib import Path
import re


@dataclass
class DataConfig:
    """Video source, temporal chunking and split definition."""

    video_dir: str = "videos"
    cache_dir: str = "cache/tokens"
    fps: float = 4.0
    chunk_frames: int = 8
    chunk_stride_frames: int = 8
    image_size: int = 256
    context_chunks: int = 3
    window_stride_chunks: int = 1
    horizon_chunks: list = field(default_factory=lambda: [1, 2])
    train_videos: list = field(default_factory=list)
    val_videos: list = field(default_factory=list)
    max_horizon_chunks: int = 8

    @property
    def chunk_seconds(self) -> float:
        return self.chunk_frames / self.fps


@dataclass
class EncoderConfig:
    kind: str = "vjepa2"  # "vjepa2" (real pretrained) or "native" (random init, tests only)
    model_id: str = "facebook/vjepa2-vitl-fpc64-256"
    revision: str = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
    allow_native_fallback: bool = False  # real runs must refuse a silent random fallback
    device: str = "cuda"
    batch_clips: int = 4

    @property
    def identity(self) -> tuple:
        """Cache identity: the native backend ignores the pretrained model fields."""
        if self.kind == "native":
            return ("native", "native")
        return (self.model_id, self.revision)


@dataclass
class ModelConfig:
    d_world: int = 256
    d_hidden: int = 512
    slots: int = 64
    heads: int = 4
    projection_seed: int = 20261006
    a_max: float = 1.0
    damping_init: float = 0.5
    substep_seconds: float = 0.25
    max_substeps: int = 256
    logvar_min: float = -8.0
    logvar_max: float = 6.0
    anchor_bandwidth: float = 2.0
    velocity_write: float = 0.5
    kl_beta: float = 0.01
    w_horizon: float = 1.0
    w_prior: float = 0.5
    w_present: float = 0.5


@dataclass
class TrainConfig:
    seed: int = 0
    batch_windows: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_steps: int = 3000
    grad_clip_norm: float = 5.0
    eval_interval: int = 250
    eval_batches: int = 8
    max_wall_seconds: float = 1800.0
    device: str = "cuda"
    resume_from: str = ""
    log_interval: int = 25


@dataclass
class DecoderConfig:
    """Architecture of the optional RGB decoder.

    The decoder maps projected world latents ``(B, P, d_world)`` of one chunk,
    with their 2D patch layout, to one RGB keyframe ``(B, 3, image_size,
    image_size)`` in ``[0, 1]``. It is trained separately from the world model
    (see :class:`DecoderTrainConfig`) and is never part of the dynamics.

    ``kind`` selects the component that builds the network (``"conv"``, the
    built-in convolutional upsampler, by default) and ``options`` carries
    architecture-specific settings for it. Both fields were appended *after* the
    original ones, so a positional ``DecoderConfig(128, 128, [1, 2, 2], 2, 1)``
    keeps its exact meaning.

    ``image_size`` is the output edge and belongs to every architecture: the
    patch grid comes from the world checkpoint, the output must be a whole
    multiple of it, and for the conv architecture ``image_size / grid_side``
    must additionally be a power of two so its upsampling stages are exact. The
    remaining fields (``base_channels``, ``channel_multipliers``,
    ``stem_blocks``, ``blocks_per_stage``) are **conv-only**; a custom ``kind``
    must leave them at their defaults and describe itself through ``options``.
    Defaults are a compact ~1.5M parameter conv network at a 16x16 patch grid
    and a 128 pixel output.
    """

    image_size: int = 128
    base_channels: int = 128
    channel_multipliers: list = field(default_factory=lambda: [1, 2, 2])
    stem_blocks: int = 2         # convolutions at patch-grid resolution (conv only)
    blocks_per_stage: int = 1    # convolutions after each x2 upsample (conv only)
    kind: str = "conv"           # registered decoder component; "conv" is the built-in
    options: dict = field(default_factory=dict)   # kind-specific, JSON-compatible


@dataclass
class DecoderTrainConfig:
    """Optimisation settings for the separate decoder training path."""

    seed: int = 0
    batch_windows: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_steps: int = 2000
    grad_clip_norm: float = 5.0
    eval_interval: int = 100
    eval_batches: int = 4
    log_interval: int = 25
    max_wall_seconds: float = 1800.0
    l1_weight: float = 1.0
    edge_weight: float = 0.1     # small finite-difference edge term; 0 disables it
    frame_cache_videos: int = 2  # per-video RGB frames held in RAM at most
    target_cache_dir: str = ""   # optional on-disk cache of decoder target keyframes


@dataclass
class PerformanceConfig:
    """Optional accelerations. Every default is the reference FP32 behaviour.

    Nothing here changes results by itself: a run with the defaults does exactly what
    earlier versions did. Each option is explicit, recorded in the checkpoints that
    were produced with it, and refused when the platform cannot honour it. An
    explicitly allowed compile setup fallback warns and records the reference path.

    ``precision`` applies to model *compute* only: the persistent state stays
    float32, clocks stay float64, and the Gaussian NLL/KL, the projection
    standardisation and every reported statistic are evaluated in float32, so a
    metric keeps its meaning when the precision changes. ``bfloat16`` is offered
    instead of float16 because it needs no loss scaler.

    The video encoder deliberately stays float32 and is not configurable here: its
    precision would change token values and therefore the cache identity and the
    decoder compatibility record, which is not worth invalidating every existing
    cache for.
    """

    precision: str = "float32"            # "float32" | "bfloat16"
    anchor_attention: str = "reference"   # "reference" (returns weights) | "sdpa"
    fused_optimizer: bool = False         # torch fused AdamW; CUDA only
    compile: bool = False                 # torch.compile on selected pure hot paths
    pin_memory: bool = False              # pinned host buffers + non-blocking copies
    non_blocking: bool = False            # async H2D; implied by pin_memory on CUDA
    compile_scope: str = "predictor"      # predictor | training_blocks
    compile_fallback: bool = False        # allow explicit reference fallback at setup only


@dataclass
class RunConfig:
    name: str = "run"
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    decoder_train: DecoderTrainConfig = field(default_factory=DecoderTrainConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        data, model, train = self.data, self.model, self.train
        if not (data.fps > 0 and math.isfinite(data.fps)):
            raise ValueError("data.fps must be positive and finite")
        if data.chunk_frames < 2 or data.chunk_frames % 2:
            raise ValueError("data.chunk_frames must be an even number >= 2 (tubelet pairing)")
        if data.chunk_stride_frames < data.chunk_frames:
            raise ValueError(
                "data.chunk_stride_frames must be >= data.chunk_frames: overlapping chunks would "
                "make a target chunk overlap the anchor it is supposed to be predicted from"
            )
        if data.image_size < 32:
            raise ValueError("data.image_size is too small")
        if data.context_chunks < 2:
            raise ValueError("data.context_chunks must be >= 2 to have a history")
        if data.window_stride_chunks < 1:
            raise ValueError("data.window_stride_chunks must be >= 1")
        if not data.horizon_chunks:
            raise ValueError("data.horizon_chunks must not be empty")
        for horizon in data.horizon_chunks:
            if type(horizon) is not int or horizon < 1:
                raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
            if horizon > data.max_horizon_chunks:
                raise ValueError(
                    f"horizon {horizon} exceeds data.max_horizon_chunks {data.max_horizon_chunks}"
                )
        if type(model.heads) is not int or model.heads < 1:
            raise ValueError("model.heads must be an integer >= 1")
        if model.d_world % model.heads:
            raise ValueError("model.d_world must be divisible by model.heads")
        if type(model.d_world) is not int or model.d_world < 1:
            raise ValueError("model.d_world must be a positive integer")
        if type(model.slots) is not int or model.slots < 1:
            raise ValueError("model.slots must be a positive integer")
        side = int(round(math.sqrt(model.slots)))
        if side * side != model.slots:
            raise ValueError("model.slots must be a perfect square (anchor grid)")
        if side > int(round(math.sqrt(256))):
            raise ValueError("model.slots must not exceed the available camera-latent grid")
        if not (math.isfinite(model.damping_init) and 0.0 < model.damping_init <= 2.0):
            raise ValueError("model.damping_init must be finite and lie in (0, 2] for stability")
        if not (model.substep_seconds > 0 and math.isfinite(model.substep_seconds)):
            raise ValueError("model.substep_seconds must be positive and finite")
        if model.substep_seconds > 0.5:
            raise ValueError(
                "model.substep_seconds must be <= 0.5: the damped integrator is only guaranteed "
                "stable for small steps"
            )
        for name in ("a_max", "anchor_bandwidth", "velocity_write"):
            value = getattr(model, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"model.{name} must be finite and non-negative")
        if not (math.isfinite(model.a_max) and model.a_max > 0):
            raise ValueError("model.a_max must be finite and positive")
        for name in ("logvar_min", "logvar_max"):
            if not math.isfinite(getattr(model, name)):
                raise ValueError(f"model.{name} must be finite")
        if model.max_substeps < 1:
            raise ValueError("model.max_substeps must be >= 1")
        maximum_delta = max(data.horizon_chunks) * data.chunk_seconds
        needed = math.ceil(maximum_delta / model.substep_seconds)
        if needed > model.max_substeps:
            raise ValueError(
                f"the longest horizon needs {needed} substeps but model.max_substeps is "
                f"{model.max_substeps}"
            )
        if model.logvar_min >= model.logvar_max:
            raise ValueError("model.logvar_min must be below model.logvar_max")
        for name in ("kl_beta", "w_horizon", "w_prior", "w_present"):
            value = getattr(model, name)
            if not (value >= 0 and math.isfinite(value)):
                raise ValueError(f"model.{name} must be a non-negative finite number")
        if model.w_horizon + model.w_prior + model.w_present <= 0:
            raise ValueError("at least one objective weight must be positive")
        for name in ("learning_rate", "grad_clip_norm", "max_wall_seconds"):
            value = getattr(train, name)
            if not (value > 0 and math.isfinite(value)):
                raise ValueError(f"train.{name} must be positive and finite")
        if train.batch_windows < 1 or train.max_steps < 1 or train.eval_batches < 1:
            raise ValueError("batch_windows, max_steps and eval_batches must be >= 1")
        if train.eval_interval < 1:
            raise ValueError("train.eval_interval must be >= 1")
        overlap = set(data.train_videos) & set(data.val_videos)
        if data.train_videos or data.val_videos:
            if overlap:
                raise ValueError(f"train and validation videos overlap: {sorted(overlap)}")
            if not data.train_videos or not data.val_videos:
                raise ValueError("both train_videos and val_videos must be given, or neither")
        self.validate_decoder()
        self.validate_performance()

    def validate_performance(self) -> None:
        """Acceleration settings, checked before a run starts."""
        validate_performance_config(self.performance)

    def validate_decoder(self) -> None:
        """Decoder sections, checked even when no decoder is trained.

        Delegates to :func:`validate_decoder_config`, which is also called by the
        decoder module itself, so the direct API (``build_decoder``/``load_decoder``)
        enforces the same rules as a configuration file.
        """
        validate_decoder_config(self.decoder, self.decoder_train)


    def save(self, path) -> None:
        path = Path(path)  # accepts str or Path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path) -> "RunConfig":
        path = Path(path)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {"name", "data", "encoder", "model", "train", "decoder", "decoder_train",
                 "performance"}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        # decoder and performance sections are optional: a config written before they
        # existed loads unchanged and keeps the reference behaviour
        config = cls(
            name=payload.get("name", "run"),
            data=DataConfig(**payload["data"]),
            encoder=EncoderConfig(**payload["encoder"]),
            model=ModelConfig(**payload["model"]),
            train=TrainConfig(**payload["train"]),
            decoder=DecoderConfig(**payload.get("decoder", {})),
            decoder_train=DecoderTrainConfig(**payload.get("decoder_train", {})),
            performance=PerformanceConfig(**payload.get("performance", {})),
        )
        config.validate()
        return config


# The built-in convolutional architecture. Every configuration written before
# decoder kinds existed describes exactly this one, which is why a missing kind is
# read as "conv" rather than as "unknown".
CONV_DECODER_KIND = "conv"

# Kind names are lowercase identifiers: they appear in configuration files, in
# checkpoint metadata and in error messages, so a stable, boring syntax beats
# accepting everything that happens to be a string.
DECODER_KIND_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")

# Architecture fields of the built-in conv decoder, in the order they were written
# before decoder kinds existed.
CONV_ARCHITECTURE_FIELDS = ("image_size", "base_channels", "channel_multipliers",
                            "stem_blocks", "blocks_per_stage")


def validate_decoder_kind(kind) -> str:
    """Validate a decoder kind name and return it unchanged.

    Pure syntax: whether the kind is *registered* is a question for the registry,
    which is why a configuration can be loaded and validated in a process that never
    imports a custom architecture.
    """
    if not isinstance(kind, str) or not DECODER_KIND_PATTERN.match(kind):
        raise ValueError(
            f"decoder.kind must be a lowercase identifier matching "
            f"{DECODER_KIND_PATTERN.pattern!r} (the built-in architecture is "
            f"{CONV_DECODER_KIND!r}), got {kind!r}"
        )
    return kind


def _check_option_value(value, path: str, seen: set) -> None:
    """One option value must survive a JSON round trip with the same types."""
    if value is None or isinstance(value, (bool, str)) or type(value) is int:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"decoder.options{path} must be finite, got {value!r}")
        return
    if isinstance(value, (dict, list)):
        if id(value) in seen:
            raise ValueError(f"decoder.options{path} contains a cycle and cannot be serialized")
        seen.add(id(value))
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in items:
            if isinstance(value, dict) and not isinstance(key, str):
                raise ValueError(f"decoder.options{path} keys must be strings, got {key!r}")
            _check_option_value(item, f"{path}['{key}']" if isinstance(value, dict)
                                else f"{path}[{key}]", seen)
        seen.discard(id(value))
        return
    raise ValueError(
        f"decoder.options{path} must be a JSON value (dict, list, string, finite number, "
        f"boolean or null); tuples, sets, tensors and objects are refused because they do "
        f"not survive a checkpoint round trip, got {type(value).__name__}"
    )


def validate_decoder_options(kind: str, options) -> dict:
    """Validate the kind-specific options and return a canonical JSON copy.

    Options are recorded in configuration files, checkpoints and summaries, so they
    have to survive a JSON round trip with unchanged types: plain ``dict``/``list``
    containers with string keys and finite numbers, nothing that would come back as
    something else. The returned value is that round trip, which also detaches it
    from the caller.
    """
    if not isinstance(options, dict):
        raise ValueError(
            f"decoder.options must be a dict of {kind!r} architecture settings, got "
            f"{type(options).__name__}"
        )
    _check_option_value(options, "", set())
    return json.loads(json.dumps(options))


def decoder_architecture(config: "DecoderConfig") -> dict:
    """Canonical architecture record of a decoder configuration.

    Used to compare a configuration against a checkpoint: two configurations with the
    same record build the same network, and any difference has to be resolved before
    training rather than discovered as a shape error later. The record always names
    the component kind, so a conv decoder can never be swapped for a different
    architecture that happens to share its output size.

    Architecture options live in their own ``options`` block -- namespaced, so an
    option called ``kind`` or ``grid`` is simply that architecture's business and can
    never shadow the shared identity. ``kind`` names the *implementation*, including
    its semantics: an author who changes what a kind computes must publish it under a
    new name (``my_decoder_v2``) rather than changing the meaning of the old one, see
    docs/decoder_components.md.
    """
    kind = validate_decoder_kind(config.kind)
    if kind == CONV_DECODER_KIND:
        return {
            "kind": CONV_DECODER_KIND,
            "image_size": int(config.image_size),
            "base_channels": int(config.base_channels),
            "channel_multipliers": [int(value) for value in config.channel_multipliers],
            "stem_blocks": int(config.stem_blocks),
            "blocks_per_stage": int(config.blocks_per_stage),
        }
    return {
        "kind": kind,
        "image_size": int(config.image_size),
        "options": validate_decoder_options(kind, config.options),
    }


def _conv_field_defaults() -> dict:
    """Defaults of the conv-only shape fields (what an unused field must still equal)."""
    defaults = {}
    for name in ("base_channels", "channel_multipliers", "stem_blocks", "blocks_per_stage"):
        field_ = DecoderConfig.__dataclass_fields__[name]
        value = field_.default
        if value is MISSING:
            value = field_.default_factory()
        defaults[name] = value
    return defaults


def decoder_config_snapshot(config: DecoderConfig) -> dict:
    """A JSON-safe copy of a decoder configuration, options included.

    Checkpoints, summaries and the module itself must never keep a live reference to
    a caller-owned options mapping: the copy is what makes an edited dictionary
    afterwards unable to change what was trained or recorded.
    """
    return asdict(config)


def _validate_conv_architecture(decoder: DecoderConfig) -> None:
    """Shape fields of the built-in conv architecture."""
    if type(decoder.base_channels) is not int or not 1 <= decoder.base_channels <= 2048:
        raise ValueError("decoder.base_channels must be an integer in [1, 2048]")
    if not decoder.channel_multipliers:
        raise ValueError("decoder.channel_multipliers must not be empty")
    for multiplier in decoder.channel_multipliers:
        if type(multiplier) is not int or not 1 <= multiplier <= 64:
            raise ValueError(
                f"decoder.channel_multipliers entries must be integers in [1, 64], got {multiplier!r}"
            )
    for name in ("stem_blocks", "blocks_per_stage"):
        value = getattr(decoder, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"decoder.{name} must be an integer >= 1")


def _reject_conv_fields(decoder: DecoderConfig) -> None:
    """A custom kind must leave the conv shape fields at their defaults.

    They take part neither in the identity of a custom architecture nor in the
    network it builds, so accepting a changed value would record a setting that
    silently does nothing.
    """
    used = sorted(name for name, default in _conv_field_defaults().items()
                  if getattr(decoder, name) != default)
    if used:
        names = ", ".join(f"decoder.{name}" for name in used)
        raise ValueError(
            f"{names} only configure kind {CONV_DECODER_KIND!r} and are ignored by kind "
            f"{decoder.kind!r}; describe that architecture through decoder.options instead "
            "(leave the conv fields at their defaults)"
        )


def validate_decoder_config(decoder: DecoderConfig,
                            settings: "DecoderTrainConfig | None" = None) -> None:
    """Architecture (and optionally optimiser) rules for the RGB decoder.

    Called by ``RunConfig.validate`` and by the decoder module itself, so the direct
    API (``build_decoder``/``load_decoder``) enforces the same rules as a
    configuration file. Grid-relative rules (the output covering the patch grid, and
    for conv the power-of-two upsampling factor) need the encoder's patch count, so
    they run when the module is built instead.

    The shared fields (``image_size``, ``kind``, ``options``) are checked for every
    architecture; the conv shape fields are checked for ``kind == "conv"`` and
    refused as unused for any other kind, so a configuration can never describe a
    network that is not the one that will be built.
    """
    validate_decoder_kind(decoder.kind)
    if type(decoder.image_size) is not int or not 32 <= decoder.image_size <= 2048:
        raise ValueError("decoder.image_size must be an integer in [32, 2048]")
    if decoder.image_size % 8:
        raise ValueError("decoder.image_size must be a multiple of 8")
    options = validate_decoder_options(decoder.kind, decoder.options)
    if decoder.kind == CONV_DECODER_KIND:
        if options:
            raise ValueError(
                f"decoder.options must be empty for kind {CONV_DECODER_KIND!r}: the conv "
                f"architecture is configured by base_channels, channel_multipliers, "
                f"stem_blocks and blocks_per_stage, got {sorted(options)}"
            )
        _validate_conv_architecture(decoder)
    else:
        _reject_conv_fields(decoder)
    if settings is None:
        return
    if type(settings.seed) is not int or settings.seed < 0:
        raise ValueError("decoder_train.seed must be a non-negative integer")
    for name in ("batch_windows", "max_steps", "eval_interval", "eval_batches", "log_interval",
                 "frame_cache_videos"):
        value = getattr(settings, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"decoder_train.{name} must be an integer >= 1")
    for name in ("learning_rate", "grad_clip_norm", "max_wall_seconds"):
        value = getattr(settings, name)
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"decoder_train.{name} must be positive and finite")
    if not (math.isfinite(settings.weight_decay) and settings.weight_decay >= 0):
        raise ValueError("decoder_train.weight_decay must be finite and non-negative")
    for name in ("l1_weight", "edge_weight"):
        value = getattr(settings, name)
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(f"decoder_train.{name} must be finite and non-negative")
    if settings.l1_weight + settings.edge_weight <= 0:
        raise ValueError("at least one decoder reconstruction weight must be positive")
    if not isinstance(settings.target_cache_dir, str):
        raise ValueError("decoder_train.target_cache_dir must be a string path "
                         "(empty disables the on-disk target cache)")


def validate_performance_config(performance: PerformanceConfig) -> None:
    """Reject an acceleration that the implementation cannot honour as written.

    Platform-dependent requests (fused optimiser, compile) are *not* judged here --
    that belongs to the run, which knows its device -- but an unknown or malformed
    value is refused immediately so a typo can never be ignored.
    """
    from .performance import PRECISIONS, ATTENTION_MODES

    if performance.precision not in PRECISIONS:
        raise ValueError(f"performance.precision must be one of {sorted(PRECISIONS)}, "
                         f"got {performance.precision!r}")
    if performance.anchor_attention not in ATTENTION_MODES:
        raise ValueError(f"performance.anchor_attention must be one of "
                         f"{sorted(ATTENTION_MODES)}, got {performance.anchor_attention!r}")
    if performance.compile_scope not in ("predictor", "training_blocks"):
        raise ValueError("performance.compile_scope must be predictor or training_blocks")
    for name in ("fused_optimizer", "compile", "compile_fallback", "pin_memory", "non_blocking"):
        if not isinstance(getattr(performance, name), bool):
            raise ValueError(f"performance.{name} must be a boolean")
