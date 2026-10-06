"""Architecture self-checks: the properties the model must have before any training.

Each check returns ``(name, passed, detail)`` and is also imported by the unit
test suite, so the CLI report and the tests can never drift apart. Everything
here runs on the small native encoder and a tiny model, never on the big backbone.
"""

import math

import torch
from torch import nn

from .config import ModelConfig
from .data import Chunk
from .encoder import NativeEncoder
from .model import VideoWorldModel, gaussian_nll_bits
from .predict import stream_chunks
from .world_state import WorldState

LN2 = math.log(2.0)
PATCHES = 16


def build_fixture(seed: int = 0, slots: int = 16, d_world: int = 32, patch: int = 16):
    """Small native-backed model: 4x4 spatial patches, no pretrained weights."""
    encoder = NativeEncoder(d_model=48, image_size=64, patch=patch, tubelet=2, seed=seed)
    config = ModelConfig(d_world=d_world, d_hidden=64, slots=slots, heads=4, substep_seconds=0.25,
                         max_substeps=256, anchor_bandwidth=1.0)
    model = VideoWorldModel(config, d_encoder=encoder.d_model, patches=encoder.patches,
                            chunk_seconds=2.0)
    model.eval()
    return model, encoder


def synthetic_tokens(rows: int, cols: int, pattern: str = "halves", seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randn(1, rows * cols, 48, generator=generator)
    half = (rows * cols) // 2
    if pattern == "halves":
        tokens[0, :half] += 4.0
        tokens[0, half:] -= 4.0
    elif pattern == "row_gradient":
        gradient = torch.linspace(-4.0, 4.0, rows)
        tokens[0] = tokens[0] + gradient.repeat_interleave(cols).unsqueeze(-1)
    return tokens


def check_spatial_heterogeneity() -> tuple:
    model, _ = build_fixture()
    state = model.initial_state(1, torch.device("cpu"), torch.float32)
    left = synthetic_tokens(4, 4, "halves")
    right = synthetic_tokens(4, 4, "row_gradient")
    next_left, _ = model.observe(state, left, 2.0, sample=False)
    next_right, _ = model.observe(state, right, 2.0, sample=False)
    slot_spread = float(next_left.slots[0].detach().std(dim=0).mean())
    slots_differ = not torch.allclose(next_left.slots, next_right.slots, atol=1e-6)
    mu_left = model.predict(next_left, 2.0)[0][0]
    mu_right = model.predict(next_right, 2.0)[0][0]
    output_spread = float(mu_left.detach().std(dim=0).mean())
    outputs_differ = not torch.allclose(mu_left, mu_right, atol=1e-6)
    # permuting the spatial input must change the spatial output
    permutation = torch.randperm(PATCHES, generator=torch.Generator().manual_seed(3))
    permuted, _ = model.observe(state, left[:, permutation], 2.0, sample=False)
    mu_permuted = model.predict(permuted, 2.0)[0][0]
    permutation_matters = not torch.allclose(mu_left, mu_permuted, atol=1e-6)
    passed = slot_spread > 1e-4 and slots_differ and output_spread > 1e-4 and outputs_differ and permutation_matters
    return ("spatial heterogeneity breaks slot symmetry", passed,
            f"slot_spread={slot_spread:.4f} slots_differ={slots_differ} "
            f"output_spread={output_spread:.4f} outputs_differ={outputs_differ} "
            f"permutation_changes_output={permutation_matters}")


def check_addressing_gradients() -> tuple:
    model, _ = build_fixture()
    state = model.initial_state(2, torch.device("cpu"), torch.float32)
    tokens = synthetic_tokens(4, 4, "halves")
    batch = torch.cat([tokens, tokens.flip(1)], dim=0)
    new_state, _ = model.observe(state, batch, 2.0, sample=True)
    mu, logvar, _, _ = model.predict(new_state, 2.0)
    loss = gaussian_nll_bits(model.project(tokens.expand(2, -1, -1)), mu, logvar).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    names = {
        "read_query": model.anchors.read_query.weight,
        "read_key": model.anchors.read_key.weight,
        "read_value": model.anchors.read_value.weight,
        "write_query": model.anchors.write_query.weight,
        "write_key": model.anchors.write_key.weight,
        "anchor_offset": model.anchors.anchor_offset,
        "anchor_embed": model.anchors.anchor_embed,
        "slot_init": model.slot_init,
    }
    report = {}
    for name, tensor in names.items():
        gradient = tensor.grad
        report[name] = None if gradient is None else float(gradient.abs().sum())
    finite = all(value is not None and math.isfinite(value) for value in report.values())
    nonzero = all(value > 0 for value in report.values())
    projection_frozen = model.projection.weight.grad is None and not model.projection.weight.requires_grad
    return ("read/write addressing parameters receive finite non-zero gradient", finite and nonzero and projection_frozen,
            ", ".join(f"{k}={v:.3g}" for k, v in report.items()) + f", projection_frozen={projection_frozen}")


def check_time_contract() -> tuple:
    model, _ = build_fixture()
    state = model.initial_state(1, torch.device("cpu"), torch.float32)
    tokens = synthetic_tokens(4, 4, "halves")
    advanced, _ = model.observe(state, tokens, 2.0, sample=False)
    time_ok = abs(float(advanced.time) - 2.0) < 1e-6
    rejected = []
    for bad in (1.5, 2.0, float("nan"), float("inf")):
        try:
            model.observe(advanced, tokens, bad, sample=False)
        except ValueError:
            rejected.append(bad)
    clone = model.advance(advanced, 0)
    zero_ok = clone is not advanced and torch.equal(clone.slots, advanced.slots) and float(clone.time) == float(advanced.time)
    try:
        model.advance(advanced, -1.0)
        negative_ok = False
    except ValueError:
        negative_ok = True
    passed = time_ok and len(rejected) == 4 and zero_ok and negative_ok
    return ("observation timestamps are absolute and monotonic; delta=0 returns an equal copy", passed,
            f"time={float(advanced.time):.3f} rejected={rejected} delta0_copy={zero_ok} negative_rejected={negative_ok}")


def check_deterministic_inference_and_rng() -> tuple:
    model, _ = build_fixture()
    state = model.initial_state(1, torch.device("cpu"), torch.float32)
    tokens = synthetic_tokens(4, 4, "halves")
    first, _ = model.observe(state, tokens, 2.0, sample=False)
    rng_before = torch.get_rng_state()
    second, _ = model.observe(state, tokens, 2.0, sample=False)
    rng_after = torch.get_rng_state()
    deterministic = torch.equal(first.slots, second.slots)
    rng_untouched = torch.equal(rng_before, rng_after)
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    sampled_a, _ = model.observe(state, tokens, 2.0, sample=True, generator=generator_a)
    sampled_b, _ = model.observe(state, tokens, 2.0, sample=True, generator=generator_b)
    seeded = torch.equal(sampled_a.slots, sampled_b.slots)
    differs = not torch.allclose(sampled_a.slots, first.slots, atol=1e-8)
    passed = deterministic and rng_untouched and seeded and differs
    return ("inference is deterministic and consumes no RNG; sampling is generator-scoped", passed,
            f"deterministic={deterministic} rng_untouched={rng_untouched} seeded_sample={seeded} "
            f"sample_differs_from_mean={differs}")


def check_observation_free_advance() -> tuple:
    model, _ = build_fixture()
    state = model.initial_state(1, torch.device("cpu"), torch.float32)
    frames_a = torch.randn(24, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    frames_b = frames_a.clone()
    frames_b[16:] = torch.randn(8, 3, 64, 64, generator=torch.Generator().manual_seed(2))
    encoder = NativeEncoder(d_model=48, image_size=64, patch=16, tubelet=2, seed=0)
    chunks = [Chunk("v", i, i * 8, i * 8 + 8, i * 2.0, i * 2.0 + 2.0) for i in range(3)]
    prefix_chunks = chunks[:2]
    state_a, _ = stream_chunks(model, encoder, frames_a, prefix_chunks, torch.device("cpu"))
    state_b, _ = stream_chunks(model, encoder, frames_b, prefix_chunks, torch.device("cpu"))
    same_history = torch.equal(state_a.slots, state_b.slots) and float(state_a.time) == float(state_b.time)
    future_a = model.predict(state_a, 2.0)[0]
    future_b = model.predict(state_b, 2.0)[0]
    future_matches = torch.equal(future_a, future_b)
    return ("states and predictions ignore observations that arrive later", same_history and future_matches,
            f"same_history_state={same_history} future_prediction_identical={future_matches}")


def check_units_and_aggregation() -> tuple:
    """Hand-computable fixture for the bit units and the averaging dimensions."""
    mu = torch.zeros(1, 4, 3)
    logvar = torch.zeros(1, 4, 3)
    target = torch.zeros(1, 4, 3)
    per_dim = gaussian_nll_bits(target, mu, logvar)
    expected_per_dim = (0.5 * math.log(2 * math.pi)) / LN2
    scalar = float(per_dim.mean())
    patches, dims = 4, 3
    per_event = float(per_dim.sum(dim=(1, 2)).mean())
    # One cell with a squared error of 4.0 (target 2.0, mean 0) and unit variance.
    perturbed = torch.zeros_like(target)
    perturbed[0, 0, 0] = 2.0
    offset_per_dim = float(gaussian_nll_bits(perturbed, mu, logvar).mean())
    expected_offset = expected_per_dim + (4.0 / 2.0) / (patches * dims) / LN2
    # float32 arithmetic: allow a relative tolerance well below any unit error.
    tolerance = 1e-5
    passed = (
        abs(scalar - expected_per_dim) < tolerance
        and abs(per_event - expected_per_dim * patches * dims) < tolerance
        and abs(offset_per_dim - expected_offset) < tolerance
    )
    return ("bits-per-dimension and per-event aggregation match hand computation", passed,
            f"per_dim={scalar:.6f} expected={expected_per_dim:.6f} "
            f"per_event={per_event:.4f} expected={expected_per_dim * patches * dims:.4f} "
            f"single_cell_offset={offset_per_dim:.6f} expected={expected_offset:.6f}")


def check_state_serialisation(tmp_dir=None) -> tuple:
    import tempfile
    from pathlib import Path

    model, _ = build_fixture()
    state = model.initial_state(1, torch.device("cpu"), torch.float32)
    tokens = synthetic_tokens(4, 4, "halves")
    advanced, _ = model.observe(state, tokens, 2.0, sample=False)
    directory = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp())
    path = directory / "state.pt"
    advanced.save(path)
    restored = WorldState.load(path, device=torch.device("cpu"), dtype=torch.float32)
    same = all(torch.equal(getattr(advanced, name), getattr(restored, name))
               for name in ("slots", "velocity", "time", "step"))
    mu_a = model.predict(advanced, 4.0)[0]
    mu_b = model.predict(restored, 4.0)[0]
    passed = same and torch.equal(mu_a, mu_b)
    return ("persistent state round-trips through disk exactly", passed,
            f"fields_equal={same} prediction_equal={torch.equal(mu_a, mu_b)}")


def check_frozen_components() -> tuple:
    model, encoder = build_fixture()
    frozen = {
        "projection_weight": model.projection.weight.requires_grad,
        "projection_mean": model.projection.mean.requires_grad,
        "anchor_grid": model.anchors.anchor_coordinates.requires_grad,
        "encoder_projection": encoder.projection.requires_grad,
    }
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    passed = not any(frozen.values()) and trainable == total and trainable > 0
    return ("frozen buffers stay frozen and every model parameter is trainable", passed,
            f"frozen_flags={frozen} trainable={trainable} total={total}")


CHECKS = (
    check_spatial_heterogeneity,
    check_addressing_gradients,
    check_time_contract,
    check_deterministic_inference_and_rng,
    check_observation_free_advance,
    check_units_and_aggregation,
    check_state_serialisation,
    check_frozen_components,
)


def run_selfcheck(config=None, verbose: bool = True) -> int:
    failures = 0
    for check in CHECKS:
        name, passed, detail = check()
        failures += 0 if passed else 1
        if verbose:
            print(f"[{'PASS' if passed else 'FAIL'}] {name}\n        {detail}", flush=True)
    if verbose:
        print(f"self-check: {len(CHECKS) - failures}/{len(CHECKS)} passed", flush=True)
    return 1 if failures else 0
