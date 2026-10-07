"""Optional accelerations: precision, attention kernel, optimiser, compile, transfer.

Everything in this module defaults to the reference FP32 path and is opt-in. Three
rules shape it:

- **explicit or nothing.** A request that the platform cannot honour raises
  ``PerformanceError`` unless setup-only ``compile_fallback`` is explicitly allowed.
  That fallback warns and records the reference policy; runtime failures abort.
  Effective settings are written into the checkpoint that used them.
- **numerics stay interpretable.** Reduced precision applies to model *compute*
  only. The persistent state stays float32, clocks stay float64, and every reported
  number (Gaussian NLL/KL, projection standardisation, pixel metrics) is computed in
  float32, so metrics from different precision settings mean the same thing.
- **no module surgery.** ``torch.compile`` is applied to a plain callable, never to
  a registered submodule, so ``state_dict`` keys, checkpoints and decoder
  compatibility are untouched; the compiled object is checked against the model's
  state-dict keys before it is accepted.

Nothing here imports Triton or any other optional accelerator: a platform without a
working compile backend simply gets a clear error telling the user to turn the
option off.
"""

import contextlib
import sys
import warnings

import torch

from .config import PerformanceConfig

#: Supported compute precisions. bfloat16 is offered instead of float16 because it
#: has the same exponent range as float32 and therefore needs no loss scaler.
PRECISIONS = ("float32", "bfloat16")

#: Anchor-read implementations: ``reference`` returns attention weights,
#: ``sdpa`` uses torch's fused kernel and does not.
ATTENTION_MODES = ("reference", "sdpa")

REDUCED_DTYPES = (torch.bfloat16, torch.float16)


class PerformanceError(RuntimeError):
    """Raised when a requested acceleration cannot be applied as requested."""


def settings(performance=None) -> PerformanceConfig:
    """Normalise ``None`` to the reference settings.

    Every public entry point accepts ``performance=None`` and must behave exactly like
    the reference configuration, so no caller can crash on a missing section (for
    example a data path driven by a bare dataset rather than a run configuration).
    """
    return performance if performance is not None else PerformanceConfig()


def as_device(device) -> torch.device:
    """Accept a ``torch.device``, a device string or ``None`` (CPU)."""
    return torch.device("cpu") if device is None else torch.device(device)


# -- precision ----------------------------------------------------------------
def autocast_context(performance, device):
    """Autocast context for model compute, or a no-op for the reference path."""
    if settings(performance).precision == "bfloat16":
        return torch.autocast(device_type=as_device(device).type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def math_dtype(*tensors) -> torch.dtype:
    """Dtype for probability/standardisation arithmetic.

    Reduced-precision inputs are promoted to float32 and a deliberate float64 run
    stays float64; the point is that the *reported* quantities never carry reduced
    precision, whatever the surrounding autocast region does.
    """
    dtype = None
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue
        dtype = tensor.dtype if dtype is None else torch.promote_types(dtype, tensor.dtype)
    if dtype is None:
        return torch.float32
    return torch.float64 if dtype == torch.float64 else torch.float32


@contextlib.contextmanager
def fp32_math(*tensors):
    """Run numerical-stability math outside autocast, in the promoted dtype.

    ``torch.autocast`` casts the inputs of matmul-like ops regardless of their
    explicit dtype, so promoting a tensor is not enough inside an autocast region --
    autocast has to be disabled around the computation itself. Callers cast their
    inputs with :func:`math_dtype` inside this context.
    """
    device = next((tensor.device for tensor in tensors if torch.is_tensor(tensor)), None)
    device_type = device.type if device is not None else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        yield


def stable_dtype(dtype: torch.dtype) -> torch.dtype:
    """Storage dtype for the persistent state: float32, never a reduced type.

    Under autocast the activations become bfloat16; the state is storage, not an
    activation, so it is promoted back. A deliberate float64 run is left alone.
    """
    return torch.float32 if dtype in REDUCED_DTYPES else dtype


def precision_policy(performance) -> dict:
    """The precision facts a checkpoint records."""
    return {
        "precision": performance.precision,
        "state_dtype": "float32",
        "clock_dtype": "float64",
        "loss_dtype": "float32",
    }


# -- optimiser ----------------------------------------------------------------
def build_optimizer(parameters, performance, learning_rate: float, weight_decay: float, device):
    """AdamW with the configured policy; CUDA-only fused kernel when requested."""
    device = as_device(device)
    performance = settings(performance)
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    kwargs = {"lr": learning_rate, "weight_decay": weight_decay}
    if performance.fused_optimizer:
        if device.type != "cuda":
            raise PerformanceError(
                "performance.fused_optimizer=True needs CUDA; the fused AdamW kernel has no CPU "
                f"implementation (device={device.type}). Set performance.fused_optimizer=false to "
                "use the reference optimiser."
            )
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(parameters, **kwargs)
    except (RuntimeError, TypeError) as error:      # pragma: no cover - platform dependent
        if "fused" not in kwargs:
            raise
        raise PerformanceError(
            f"this torch build cannot create a fused AdamW ({error}); set "
            "performance.fused_optimizer=false to use the reference optimiser"
        ) from error


def optimizer_policy(performance, device) -> dict:
    """The optimiser facts a checkpoint records, as actually applied."""
    device = as_device(device)
    performance = settings(performance)
    return {
        "name": "AdamW",
        "fused": bool(performance.fused_optimizer and device.type == "cuda"),
        "requested_fused": bool(performance.fused_optimizer),
        "device_type": device.type,
    }


# -- compile ------------------------------------------------------------------
class _Callable:
    """Thin callable wrapper so ``torch.compile`` never sees an ``nn.Module``."""

    def __init__(self, function):
        self._function = function

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)


#: Substrings that identify a *compiler backend* failure rather than a model bug.
BACKEND_FAILURE_MARKERS = ("backendcompilerfailed", "invalidcxxcompiler", "compiler:",
                           "inductor", "triton", "torch.compile", "dynamo")


class _GuardedCallable:
    """Wraps a compiled callable so a lazy backend failure stays a clear error.

    ``torch.compile`` is lazy: it can return successfully and only raise when the
    graph is first executed, possibly after a checkpoint has already been written.
    Backend failures are re-raised as ``PerformanceError`` with the disable advice;
    anything else is passed through untouched so a genuine model bug is never
    relabelled as a compile problem.
    """

    def __init__(self, function, label: str):
        self._function = function
        self._label = label

    def __call__(self, *args, **kwargs):
        try:
            return self._function(*args, **kwargs)
        except Exception as error:                    # noqa: BLE001 - re-raised below
            message = f"{type(error).__name__}: {error}".lower()
            if any(marker in message for marker in BACKEND_FAILURE_MARKERS):
                raise PerformanceError(
                    f"the compiled {self._label} failed at run time "
                    f"({type(error).__name__}: {error}). Set performance.compile=false to run the "
                    "reference path."
                ) from error
            raise


def apply_compile(model, performance, compile_fn=None) -> dict:
    """Apply the requested scope atomically; optional setup-only reference fallback.

    A runtime/backend or backward failure after setup always aborts. Changing the
    effective policy halfway through training could invalidate a checkpoint.
    """
    performance = settings(performance)
    model.compiled_predictor = None
    model.compiled_training = None
    try:
        status = (_apply_training_blocks(model, performance, compile_fn)
                  if performance.compile and performance.compile_scope == "training_blocks"
                  else _apply_predictor_compile(model, performance, compile_fn))
    except PerformanceError as error:
        if not performance.compile_fallback:
            raise
        model.compiled_predictor = None
        model.compiled_training = None
        status = {"compile": bool(performance.compile), "applied": False, "verified": False,
                  "state": "fallback", "targets": [], "backend": sys.platform, "reason": str(error)}
        warnings.warn(f"Compilation preflight failed; using the reference implementation: {error}",
                      RuntimeWarning, stacklevel=2)
    status["requested_scope"] = performance.compile_scope
    status["scope"] = performance.compile_scope if status["applied"] else "none"
    return status


def _apply_training_blocks(model, performance, compile_fn):
    from .compiled_training import apply_training_blocks
    return apply_training_blocks(model, performance, compile_fn or torch.compile)


def _apply_predictor_compile(model, performance, compile_fn=None) -> dict:
    """Attach a compiled copy of the model's predictor head, or report why not.

    Returns a status record with ``applied``/``reason``/``targets``. Compilation is
    applied to a plain callable, so the model's ``state_dict`` keys are unchanged;
    that is verified here and a violation is an error rather than a broken
    checkpoint. Any failure of ``torch.compile`` itself becomes a
    ``PerformanceError`` telling the user to disable the option, because a request
    that cannot be honoured must never look like it ran.

    Whether the backend can actually build the graph depends on the torch version, the
    selected backend and the platform (a missing C++ toolchain raises
    ``InvalidCxxCompiler``, for example); that is why the preflight runs a real
    execution instead of trusting the call to ``torch.compile``. Nothing here depends
    on Triton or on any custom kernel: a platform without a working backend gets the
    explicit error above and is expected to run with ``performance.compile=false``.
    """
    performance = settings(performance)
    status = {"compile": bool(performance.compile), "applied": False, "verified": False,
              "state": "disabled", "targets": [], "backend": None, "reason": None}
    if not performance.compile:
        status["reason"] = "disabled by configuration"
        return status
    compile_fn = compile_fn or torch.compile
    keys_before = set(model.state_dict())
    try:
        compiled = compile_fn(_Callable(model.predictor))
    except Exception as error:
        raise PerformanceError(
            f"performance.compile=True but torch.compile failed on this platform "
            f"({type(error).__name__}: {error}). Set performance.compile=false to run the "
            "reference path."
        ) from error
    if not callable(compiled):
        raise PerformanceError(
            "torch.compile returned a non-callable object; set performance.compile=false"
        )
    if set(model.state_dict()) != keys_before:
        # e.g. a compiled *module* would rename keys with an `_orig_mod.` prefix and
        # silently invalidate every existing checkpoint
        raise PerformanceError(
            "compiling changed the model's state_dict keys, which would break checkpoint "
            "compatibility; set performance.compile=false"
        )
    guarded = _GuardedCallable(compiled, "predictor head")
    probe = _preflight_inputs(model)
    if probe is not None:
        # torch.compile is lazy: returning successfully proves nothing. Run the head
        # once here, before any checkpoint records the option, so a backend that cannot
        # build turns into an explicit error instead of a late training crash.
        _preflight(model, guarded, probe)
        status["verified"] = True
    model.compiled_predictor = guarded
    status.update(applied=True, state="applied" if status["verified"] else "unverified",
                  targets=["predictor"],
                  reason="applied and executed" if status["verified"] else
                         "applied, execution not verified (no probe shape available)",
                  backend=sys.platform)
    return status


def _preflight_inputs(model):
    """A zero-filled feature batch shaped like the predictor's real input, or None.

    The predictor consumes one read-output vector per patch, i.e. ``(B, P, d_world)``.
    """
    config = getattr(model, "config", None)
    patches = getattr(model, "patches", None)
    if config is None or patches is None or not hasattr(model, "predictor"):
        return None
    parameter = next(model.predictor.parameters(), None)
    if parameter is None:
        return None
    return torch.zeros(1, int(patches), int(config.d_world), dtype=parameter.dtype,
                       device=parameter.device)


def _preflight(model, head, probe) -> None:
    """Execute the compiled head once, leaving the model exactly as it was found.

    The probe must be invisible: it restores the CPU RNG (and the CUDA RNG streams when
    CUDA is already in use -- ``is_initialized`` is checked so a CPU run never
    initialises a CUDA context just to snapshot it) and puts every submodule back into
    the training flag it had, instead of forcing the whole model into one mode.
    """
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = {name: module.training for name, module in model.named_modules()}
    try:
        model.eval()
        with torch.no_grad():
            head(probe)
    except PerformanceError:
        raise
    except Exception as error:                        # noqa: BLE001 - re-raised below
        raise PerformanceError(
            f"performance.compile=True but the compiled predictor head could not be executed "
            f"({type(error).__name__}: {error}). Set performance.compile=false to run the "
            "reference path."
        ) from error
    finally:
        # the probe must not consume randomness or leave any module in another mode
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        for name, module in model.named_modules():
            if name in modes:
                module.train(modes[name])


# -- host/device transfer -----------------------------------------------------
def should_pin(performance, device) -> bool:
    """Pinned staging buffers only help (and only exist) for CUDA transfers.

    ``performance`` may be ``None`` (the reference settings) and ``device`` may be a
    string, so every data path can call this without normalising first.
    """
    device = as_device(device)
    return bool((settings(performance).pin_memory or settings(performance).non_blocking)
                and device.type == "cuda")


def to_device(tensor: torch.Tensor, device, performance=None, dtype=None) -> torch.Tensor:
    """Move a CPU tensor to ``device``, optionally through pinned memory.

    Pinning is a staging detail: the returned tensor is identical either way, so a
    run with and without it must produce the same numbers. ``performance=None`` means
    the reference settings (no pinning) rather than an error.
    """
    device = as_device(device)
    if dtype is not None:
        tensor = tensor.to(dtype)
    if not should_pin(performance, device):
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


# -- checkpoint policy --------------------------------------------------------
REFERENCE_POLICY = {
    "precision": "float32",
    "state_dtype": "float32",
    "clock_dtype": "float64",
    "loss_dtype": "float32",
    "anchor_attention": "reference",
    "fused": False,
    "compile": False,
    "compile_scope": "none",
    "optimizer_name": "AdamW",
}

#: The semantic entries a resume must not silently change: how the model computes and
#: how the optimiser updates it. Everything else a checkpoint records (device, pinned
#: transfers, verbose compile status, dtype bookkeeping) describes *where* or *how
#: fast* a run happened, and must not block a legitimate cross-device resume.
COMPARABLE_POLICY_KEYS = ("precision", "anchor_attention", "fused", "compile", "compile_scope")


def comparable_policy(policy: dict | None) -> dict:
    """The semantic subset of a policy record that a resume must not change.

    Both the flat ``fused`` flag and the one nested in ``optimizer`` are picked up, so
    a decoder policy (which records the optimiser as a block) is compared as strictly
    as a world policy.
    """
    if not policy:
        return {}
    flat = {key: policy[key] for key in COMPARABLE_POLICY_KEYS if key in policy}
    if "compile" in flat:
        flat["compile_scope"] = policy.get("compile_scope", "predictor") if flat["compile"] else "none"
    optimizer = policy.get("optimizer")
    if isinstance(optimizer, dict):
        if "name" in optimizer:
            flat["optimizer_name"] = optimizer["name"]
        if "fused" in optimizer:
            flat["fused"] = bool(optimizer["fused"])
    return flat


def resume_policy_error(recorded: dict | None, current: dict, what: str) -> str | None:
    """Compare a checkpoint's recorded policy with the settings in force.

    Only the scalar policy entries are compared -- precision, attention kernel,
    fused-optimiser flag, compile flag, optimiser name -- so recorded device details
    do not block a legitimate cross-device resume. A checkpoint that predates the
    record is treated as the reference policy: it can be resumed with reference
    settings only, never with an acceleration it never recorded.
    """
    current = comparable_policy(current)
    if recorded is None:
        for key, value in REFERENCE_POLICY.items():
            if key in current and current[key] != value:
                return (f"this checkpoint does not record its {what} policy (it predates the "
                        f"setting); it was trained with the reference policy, so resuming with "
                        f"{what}.{key}={current[key]!r} would not continue the same run")
        return None
    # a key the record does not mention is treated as the reference value, so an
    # incomplete record can never silently accept an acceleration it cannot account for
    baseline = {**REFERENCE_POLICY, **comparable_policy(recorded)}
    for key, value in current.items():
        if key in baseline and baseline[key] != value:
            return (f"{what}.{key} differs: checkpoint={baseline[key]!r}, current={value!r}; "
                    "resuming with different settings would silently change the run")
    return None


def require_matching_policy(recorded: dict | None, current: dict, what: str) -> None:
    """Raise ``PerformanceError`` when a resume would change the acceleration policy."""
    problem = resume_policy_error(recorded, current, what)
    if problem:
        raise PerformanceError(problem)


def training_policy(performance, model, device, compile_status: dict | None = None) -> dict:
    """The acceleration policy actually in force for one training run.

    The values are *effective*, not requested: the fused flag is only true on CUDA and
    ``pin_memory`` only true where pinned transfers exist, so a checkpoint never claims
    an acceleration that the run did not actually use.
    """
    performance = settings(performance)
    return {
        "precision": performance.precision,
        "anchor_attention": getattr(model, "anchor_attention", "reference"),
        "fused": optimizer_policy(performance, device)["fused"],
        "compile": bool((compile_status or {}).get("applied", False)),
        "compile_scope": (compile_status or {}).get("scope", "predictor" if (compile_status or {}).get("applied") else "none"),
        "pin_memory": should_pin(performance, device),
    }
