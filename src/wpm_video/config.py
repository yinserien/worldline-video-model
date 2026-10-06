"""Run configuration: data, encoder, model, training and optional decoder settings.

Plain dataclasses with JSON round-tripping plus strict validation. Every field
here is used by the implementation, and ``validate`` refuses combinations that
would silently change the meaning of a run (bad dimensions, illegal horizons,
overlapping splits, unstable integration).

The ``decoder`` and ``decoder_train`` sections configure the *optional* RGB
decoder. They are absent from every configuration written before the decoder
existed and default to valid values, so an old ``config.json`` loads unchanged
and the world-model commands behave exactly as before.
"""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


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

    ``image_size`` is the output edge; the patch grid comes from the world
    checkpoint, and ``image_size / grid_side`` must be a power of two so the
    upsampling stages are exact. Defaults are a compact ~1.5M parameter network
    at a 16x16 patch grid and a 128 pixel output.
    """

    image_size: int = 128
    base_channels: int = 128
    channel_multipliers: list = field(default_factory=lambda: [1, 2, 2])
    stem_blocks: int = 2         # convolutions at patch-grid resolution
    blocks_per_stage: int = 1    # convolutions after each x2 upsample


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


@dataclass
class RunConfig:
    name: str = "run"
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    decoder_train: DecoderTrainConfig = field(default_factory=DecoderTrainConfig)

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
        known = {"name", "data", "encoder", "model", "train", "decoder", "decoder_train"}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        # decoder sections are optional: a config written before the decoder existed
        # loads unchanged and keeps the behaviour of the world-model commands
        config = cls(
            name=payload.get("name", "run"),
            data=DataConfig(**payload["data"]),
            encoder=EncoderConfig(**payload["encoder"]),
            model=ModelConfig(**payload["model"]),
            train=TrainConfig(**payload["train"]),
            decoder=DecoderConfig(**payload.get("decoder", {})),
            decoder_train=DecoderTrainConfig(**payload.get("decoder_train", {})),
        )
        config.validate()
        return config


def validate_decoder_config(decoder: DecoderConfig,
                            settings: "DecoderTrainConfig | None" = None) -> None:
    """Architecture (and optionally optimiser) rules for the RGB decoder.

    Called by ``RunConfig.validate`` and by the decoder module itself, so the direct
    API (``build_decoder``/``load_decoder``) enforces the same rules as a
    configuration file. The grid-relative check (``image_size`` divided by the patch
    grid must be a power of two) needs the encoder's patch count, so it runs in
    ``build_decoder`` instead.
    """
    if type(decoder.image_size) is not int or not 32 <= decoder.image_size <= 2048:
        raise ValueError("decoder.image_size must be an integer in [32, 2048]")
    if decoder.image_size % 8:
        raise ValueError("decoder.image_size must be a multiple of 8")
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


def decoder_architecture(config: "DecoderConfig") -> dict:
    """The architecture fields that define a decoder's parameter shapes.

    Used to compare a configuration against a checkpoint: two configurations with the
    same dictionary build the same network, and any difference has to be resolved
    before training rather than discovered as a shape error later.
    """
    return {
        "image_size": int(config.image_size),
        "base_channels": int(config.base_channels),
        "channel_multipliers": [int(value) for value in config.channel_multipliers],
        "stem_blocks": int(config.stem_blocks),
        "blocks_per_stage": int(config.blocks_per_stage),
    }
