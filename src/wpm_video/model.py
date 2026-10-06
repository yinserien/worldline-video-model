"""The video world model: spatially anchored persistent state, innovation, dynamics.

Data flow for one observed chunk whose frame interval ends at ``end_seconds``:

    state'  = advance(state, end_seconds - state.time)     # observation-free prediction
    prior   = N(mu_p, var_p | read(state'))                # what the state expects to see
    post    = N(mu_q, var_q | read(state'), observation)   # what the camera actually shows
    z       ~ post                    (training) / mu_q    (inference)
    innovation = z - mu_p
    state'' = write(state', innovation) with position aware content addressing

Only the innovation may change the persistent state, so no raw video feature can
bypass the correction path.

Spatial collapse is prevented by construction: slots start from distinct learned
values, each slot carries its own camera-latent anchor coordinate, and both read
and write multiply content similarity with a position term, so a patch addresses
the anchors near it and different image regions write to different anchors.
Anchors are camera-latent addresses only; nothing here is a 3D reconstruction.

Prediction is residual around the model's own present estimate: the anchor term
is the read at delta 0 and the predictor head is zero initialized, so training
starts exactly at the persistence behaviour and only has to learn the change.
"""

import math

import torch
from torch import nn

from .config import ModelConfig
from .world_state import WorldState

LN2 = math.log(2.0)


def gaussian_kl_bits(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
    """KL(q||p) between diagonal Gaussians; returns per-dimension bits, shape (..., D)."""
    nats = 0.5 * (logvar_p - logvar_q + (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp() - 1.0)
    return nats / LN2


def gaussian_nll_bits(target, mu, logvar) -> torch.Tensor:
    """-log2 N(target | mu, var); returns per-dimension bits, shape (..., D)."""
    nats = 0.5 * (logvar + (target - mu) ** 2 / logvar.exp()) + 0.5 * math.log(2 * math.pi)
    return nats / LN2


def grid_coordinates(rows: int, cols: int) -> torch.Tensor:
    """Patch centre coordinates normalised to [-1, 1]^2, ordered row major."""
    ys = torch.linspace(-1.0, 1.0, rows)
    xs = torch.linspace(-1.0, 1.0, cols)
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)
    return grid.reshape(-1, 2)


class LatentProjection(nn.Module):
    """Frozen random semi-orthogonal projection plus train-fitted standardisation.

    Buffers, never trained, fitted on training tokens only. This is the stable
    target space the frozen encoder provides; it cannot collapse with the
    predictor because no gradient ever reaches it.
    """

    def __init__(self, d_in: int, d_out: int, seed: int):
        super().__init__()
        if d_out > d_in:
            raise ValueError(f"d_world ({d_out}) must not exceed the encoder width ({d_in})")
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(d_in, d_out, generator=generator)
        q, _ = torch.linalg.qr(weight)
        self.register_buffer("weight", q[:, :d_out].contiguous())
        self.register_buffer("mean", torch.zeros(d_out))
        self.register_buffer("std", torch.ones(d_out))

    @torch.no_grad()
    def fit_standardisation(self, tokens: torch.Tensor, eps: float = 1e-3) -> None:
        flat = tokens.reshape(-1, tokens.shape[-1]).to(self.weight.device, self.weight.dtype) @ self.weight
        self.mean.copy_(flat.mean(dim=0))
        self.std.copy_(flat.std(dim=0).clamp_min(eps))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return ((tokens @ self.weight) - self.mean) / self.std


class SpatialAnchors(nn.Module):
    """Position-aware content addressing over persistent spatial anchor slots."""

    def __init__(self, d_model: int, slots: int, grid: tuple[int, int], heads: int, bandwidth: float):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_world must be divisible by heads")
        self.d_model = d_model
        self.slots = slots
        self.heads = heads
        self.head_dim = d_model // heads
        self.register_buffer("patch_coordinates", grid_coordinates(*grid))
        # Learned per-patch query embedding; the raw 2D coordinates only enter the
        # position bias, which keeps the addressing geometric rather than an index.
        self.patch_embed = nn.Parameter(torch.randn(grid[0] * grid[1], d_model) * 0.02)
        anchor_grid = grid_coordinates(*self._anchor_grid_shape(slots))
        self.register_buffer("anchor_coordinates", anchor_grid)
        self.anchor_offset = nn.Parameter(torch.zeros(slots, 2))
        self.anchor_embed = nn.Parameter(torch.randn(slots, d_model) * 0.02)
        self.read_query = nn.Linear(d_model, d_model)
        self.read_key = nn.Linear(d_model, d_model)
        self.read_value = nn.Linear(d_model, d_model)
        self.write_query = nn.Linear(d_model, d_model)
        self.write_key = nn.Linear(d_model, d_model)
        bandwidth = max(bandwidth, 1e-3)
        self.log_bandwidth = nn.Parameter(torch.tensor(math.log(math.expm1(bandwidth))))

    @staticmethod
    def _anchor_grid_shape(slots: int) -> tuple[int, int]:
        side = int(round(math.sqrt(slots)))
        if side * side != slots:
            raise ValueError(f"slots must be a perfect square for the anchor grid, got {slots}")
        return side, side

    @property
    def bandwidth(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_bandwidth) + 1e-3

    def position_bias(self, coordinates: torch.Tensor) -> torch.Tensor:
        """(N, 2) query coordinates -> (N, M) log-space position preference."""
        anchors = self.anchor_coordinates + torch.tanh(self.anchor_offset)
        distance = (coordinates.unsqueeze(1) - anchors.unsqueeze(0)).pow(2).sum(dim=-1)
        return -self.bandwidth * distance

    def _split_heads(self, tensor: torch.Tensor, batch: int, length: int) -> torch.Tensor:
        return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def read(self, slots: torch.Tensor, queries: torch.Tensor, coordinates: torch.Tensor):
        batch, length, _ = queries.shape
        q = self._split_heads(self.read_query(queries), batch, length)
        k = self._split_heads(self.read_key(slots) + self.anchor_embed, batch, slots.shape[1])
        v = self._split_heads(self.read_value(slots), batch, slots.shape[1])
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim) + self.position_bias(coordinates)
        attention = torch.softmax(scores, dim=-1)
        out = (attention @ v).transpose(1, 2).reshape(batch, length, self.d_model)
        return out, attention

    def write(self, slots: torch.Tensor, tokens: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Route token increments into anchors by content similarity and position."""
        batch, length, _ = tokens.shape
        q = self._split_heads(self.write_query(tokens), batch, length)
        k = self._split_heads(self.write_key(slots) + self.anchor_embed, batch, slots.shape[1])
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim) + self.position_bias(coordinates)
        attention = torch.softmax(scores, dim=-1).mean(dim=1)          # average heads -> (B, N, M)
        routed = attention.transpose(1, 2) @ tokens                    # (B, M, D)
        mass = attention.sum(dim=1).clamp_min(1e-6).unsqueeze(-1)      # (B, M, 1)
        return routed / mass


class VideoWorldModel(nn.Module):
    def __init__(self, config: ModelConfig, d_encoder: int, patches: int, chunk_seconds: float):
        super().__init__()
        # Validate before allocating anything: heads=0 or a bad dimension must raise
        # ValueError here, never a ZeroDivisionError from a later modulo or sqrt.
        self.config = config
        self.patches = patches
        self.chunk_seconds = chunk_seconds
        self.validate_configuration()
        d = config.d_world
        grid_side = int(round(math.sqrt(patches)))
        if grid_side * grid_side != patches:
            raise ValueError(f"patches must be a perfect square, got {patches}")
        self.grid = (grid_side, grid_side)
        self.projection = LatentProjection(d_encoder, d, config.projection_seed)
        self.anchors = SpatialAnchors(d, config.slots, self.grid, config.heads, config.anchor_bandwidth)
        self.slot_init = nn.Parameter(torch.randn(config.slots, d) * 0.02)
        self.prior = nn.Sequential(nn.Linear(d, config.d_hidden), nn.SiLU(), nn.Linear(config.d_hidden, 2 * d))
        self.posterior = nn.Sequential(
            nn.Linear(2 * d, config.d_hidden), nn.SiLU(), nn.Linear(config.d_hidden, 2 * d)
        )
        self.present = nn.Sequential(nn.Linear(d, config.d_hidden), nn.SiLU(), nn.Linear(config.d_hidden, 2 * d))
        self.horizon = nn.Sequential(nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d))
        self.predictor = nn.Sequential(nn.Linear(d, config.d_hidden), nn.SiLU(), nn.Linear(config.d_hidden, 2 * d))
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        self.acceleration = nn.Sequential(
            nn.Linear(2 * d, config.d_hidden), nn.SiLU(), nn.Linear(config.d_hidden, d)
        )
        nn.init.zeros_(self.acceleration[-1].weight)
        nn.init.zeros_(self.acceleration[-1].bias)
        damping = min(max(config.damping_init, 1e-3), 1.999)
        self.damping_raw = nn.Parameter(torch.tensor(math.log(damping / (2.0 - damping))))
        self.velocity_write = nn.Parameter(torch.tensor(config.velocity_write))

    def validate_configuration(self) -> None:
        """Constructor-level contract, checked before any tensor is allocated."""
        config = self.config
        if not (math.isfinite(self.chunk_seconds) and self.chunk_seconds > 0):
            raise ValueError("chunk_seconds must be finite and positive")
        if type(self.patches) is not int or self.patches < 1:
            raise ValueError("patches must be a positive integer")
        for name in ("d_world", "d_hidden", "slots"):
            value = getattr(config, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(config.heads) is not int or config.heads < 1:
            raise ValueError("heads must be an integer >= 1")
        if config.d_world % config.heads:
            raise ValueError("d_world must be divisible by heads")
        if not (math.isfinite(config.substep_seconds) and 0.0 < config.substep_seconds <= 0.5):
            raise ValueError("substep_seconds must be finite and in (0, 0.5] for a stable integrator")
        if type(config.max_substeps) is not int or config.max_substeps < 1:
            raise ValueError("max_substeps must be an integer >= 1")
        if not (math.isfinite(config.damping_init) and 0.0 < config.damping_init <= 2.0):
            raise ValueError("damping_init must be finite and lie in (0, 2]")
        for name in ("a_max", "anchor_bandwidth", "velocity_write"):
            value = getattr(config, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("logvar_min", "logvar_max"):
            if not math.isfinite(getattr(config, name)):
                raise ValueError(f"{name} must be finite")
        if config.logvar_min >= config.logvar_max:
            raise ValueError("logvar_min must be below logvar_max")

    # -- helpers -------------------------------------------------------------
    @property
    def patch_coordinates(self) -> torch.Tensor:
        return self.anchors.patch_coordinates

    def project(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(tokens)

    def initial_state(self, batch_size: int, device=None, dtype=None) -> WorldState:
        reference = self.slot_init
        device = device or reference.device
        dtype = dtype or reference.dtype
        slots = self.slot_init.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        return WorldState(
            slots=slots,
            velocity=torch.zeros_like(slots),
            # the clock is float64 metadata even when the state dtype is float32
            time=torch.zeros(batch_size, device=device, dtype=torch.float64),
            step=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def _clamp_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        return logvar.clamp(self.config.logvar_min, self.config.logvar_max)

    def _queries(self, batch: int, device, dtype, delta_seconds=None) -> torch.Tensor:
        queries = self.anchors.patch_embed.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)
        if delta_seconds is None:
            return queries
        delta = torch.as_tensor(delta_seconds, device=device, dtype=dtype).reshape(-1)
        if delta.numel() == 1:
            delta = delta.expand(batch)
        if delta.numel() != batch:
            raise ValueError(f"delta_seconds must be scalar or have {batch} entries, got {delta.numel()}")
        features = torch.stack([delta, torch.log1p(delta)], dim=-1)
        return queries + self.horizon(features).unsqueeze(1)

    # -- observation path ----------------------------------------------------
    def observe(self, state: WorldState, tokens: torch.Tensor, end_seconds, sample: bool | None = None,
                generator: torch.Generator | None = None):
        """Correct the state with one observation whose frame interval ends at ``end_seconds``.

        The state is first advanced to that timestamp with no observation, so the
        prior is a genuine prediction of what is about to be seen.

        ``sample`` defaults to the module's training flag: inference (``model.eval()``)
        uses the deterministic posterior mean, training draws a reparameterized
        sample. A generator can be supplied to make the draw reproducible without
        touching the global RNG.
        """
        if sample is None:
            sample = self.training
        end = torch.as_tensor(end_seconds, device=state.time.device, dtype=torch.float64)
        if end.dim() == 0:
            end = end.expand(state.batch_size())
        if not bool(torch.isfinite(end).all()):
            raise ValueError("Observation end time must be finite")
        delta = end - state.time
        if bool((delta < 0).any()):
            raise ValueError("Observation end time must not move backwards relative to the state")
        if bool((delta == 0).any()):
            raise ValueError("Observation end time must be strictly later than the state time")
        advanced = self.advance(state, delta)
        observation = self.project(tokens)
        batch = observation.shape[0]
        coordinates = self.anchors.patch_coordinates.to(observation.device, observation.dtype)
        history, _ = self.anchors.read(advanced.slots, self._queries(batch, observation.device,
                                                                    observation.dtype), coordinates)
        mu_prior, logvar_prior = self.prior(history).chunk(2, dim=-1)
        mu_post, logvar_post = self.posterior(torch.cat([history, observation], dim=-1)).chunk(2, dim=-1)
        logvar_prior = self._clamp_logvar(logvar_prior)
        logvar_post = self._clamp_logvar(logvar_post)
        if sample:
            noise = torch.randn(mu_post.shape, device=mu_post.device, dtype=mu_post.dtype,
                                generator=generator)
            code = mu_post + torch.exp(0.5 * logvar_post) * noise
        else:
            code = mu_post
        innovation = code - mu_prior
        increment = self.anchors.write(advanced.slots, innovation, coordinates)
        new_state = WorldState(
            slots=advanced.slots + increment,
            velocity=advanced.velocity + self.velocity_write * increment,
            time=end,
            step=advanced.step + 1,
        )
        diagnostics = {
            "kl_bits_per_dim": gaussian_kl_bits(mu_post, logvar_post, mu_prior, logvar_prior).mean(dim=-1),
            "prior_nll_bits_per_dim": gaussian_nll_bits(observation, mu_prior, logvar_prior).mean(dim=-1),
            "innovation_rms": innovation.pow(2).mean(dim=(1, 2)).sqrt(),
            "increment_rms": increment.pow(2).mean(dim=(1, 2)).sqrt(),
            "sampled": sample,
        }
        return new_state, diagnostics

    # -- observation-free dynamics ------------------------------------------
    def advance(self, state: WorldState, delta_seconds) -> WorldState:
        """Move the state forward in real time without reading any observation."""
        if isinstance(delta_seconds, torch.Tensor):
            if delta_seconds.numel() not in (1, state.batch_size()):
                raise ValueError(
                    f"advance expects a scalar delta or one delta per state row "
                    f"({state.batch_size()}), got {delta_seconds.numel()}"
                )
            if not bool(torch.isfinite(delta_seconds).all()):
                raise ValueError("advance requires finite time deltas")
            if bool((delta_seconds < 0).any()):
                raise ValueError("advance requires non-negative time deltas")
            if bool((delta_seconds == 0).all()):
                return state.clone()
            maximum = float(delta_seconds.max())
        else:
            if not math.isfinite(delta_seconds):
                raise ValueError("advance requires a finite time delta")
            if delta_seconds < 0:
                raise ValueError("advance requires a non-negative time delta")
            if delta_seconds == 0:
                return state.clone()
            maximum = float(delta_seconds)
        steps = int(math.ceil(maximum / self.config.substep_seconds))
        if steps > self.config.max_substeps:
            raise ValueError(
                f"advance would need {steps} substeps, above the configured limit "
                f"{self.config.max_substeps}; reduce the delta or raise substep_seconds"
            )
        step_seconds = delta_seconds / steps
        if torch.is_tensor(step_seconds) and step_seconds.dim() == 1:
            step_seconds = step_seconds.reshape(-1, 1, 1)  # broadcast per batch row
        # the integrator runs in the state dtype; the clock stays float64
        step_seconds = torch.as_tensor(step_seconds, dtype=state.slots.dtype)
        damping = torch.sigmoid(self.damping_raw) * 2.0
        slots, velocity = state.slots, state.velocity
        for _ in range(steps):
            acceleration = self.config.a_max * torch.tanh(
                self.acceleration(torch.cat([slots, velocity], dim=-1))
            )
            velocity = (1.0 - damping * step_seconds) * velocity + acceleration * step_seconds
            slots = slots + velocity * step_seconds
        return WorldState(slots, velocity, state.time + delta_seconds, state.step)

    # -- readout and multi-horizon prediction --------------------------------
    def present_estimate(self, state: WorldState):
        """The model's estimate of the latent it is currently standing on (delta = 0)."""
        batch = state.batch_size()
        coordinates = self.anchors.patch_coordinates.to(state.slots.device, state.slots.dtype)
        features, _ = self.anchors.read(state.slots, self._queries(batch, state.slots.device,
                                                                  state.slots.dtype), coordinates)
        mu, logvar = self.present(features).chunk(2, dim=-1)
        return mu, self._clamp_logvar(logvar)

    def predict(self, state: WorldState, delta_seconds):
        """Future spatial latent distribution ``delta_seconds`` after the state time.

        The prediction is the present estimate plus a learned change, so it starts
        from the persistence behaviour and only has to learn the difference.
        """
        advanced = self.advance(state, delta_seconds)
        batch = advanced.batch_size()
        coordinates = self.anchors.patch_coordinates.to(advanced.slots.device, advanced.slots.dtype)
        reference, _ = self.present_estimate(state)
        queries = self._queries(batch, advanced.slots.device, advanced.slots.dtype, delta_seconds)
        features, attention = self.anchors.read(advanced.slots, queries, coordinates)
        change, logvar = self.predictor(features).chunk(2, dim=-1)
        return reference + change, self._clamp_logvar(logvar), advanced, attention

    def reset_state(self, state: WorldState, mask: torch.Tensor) -> WorldState:
        """Reset selected batch rows to the same learned initial state a fresh video gets."""
        initial = self.initial_state(state.batch_size(), state.slots.device, state.slots.dtype)
        keep = (~mask).to(state.slots.dtype).reshape(-1, 1, 1)
        return WorldState(
            slots=state.slots * keep + initial.slots * (1 - keep),
            velocity=state.velocity * keep,
            time=state.time * keep.reshape(-1),
            step=(state.step * keep.reshape(-1).long()),
        )

    # -- persistence ---------------------------------------------------------
    def save(self, path, extra: dict | None = None) -> None:
        payload = {
            "schema_version": 2,
            "model_config": vars(self.config),
            "patches": self.patches,
            "chunk_seconds": self.chunk_seconds,
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @staticmethod
    def from_checkpoint(path, map_location="cpu"):
        payload = torch.load(path, map_location=map_location, weights_only=False)
        config = ModelConfig(**payload["model_config"])
        d_encoder = payload["state_dict"]["projection.weight"].shape[0]
        model = VideoWorldModel(config, d_encoder, payload["patches"], payload["chunk_seconds"])
        model.load_state_dict(payload["state_dict"])
        return model, payload
