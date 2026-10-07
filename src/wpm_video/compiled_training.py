"""Optional fusion of existing training blocks; no custom accelerator dependency.

Callables are prepared off-model, probed in the requested AMP policy, then attached
as one plain dict. Reference methods and state_dict remain available unchanged.
"""
import inspect
import torch

from .model import gaussian_nll_bits, gaussian_kl_bits
from .performance import PerformanceError, _GuardedCallable, autocast_context


def nll_rows(target, mu, logvar):
    return gaussian_nll_bits(target, mu, logvar).mean(dim=-1)


def kl_rows(mu_q, logvar_q, mu_p, logvar_p):
    return gaussian_kl_bits(mu_q, logvar_q, mu_p, logvar_p).mean(dim=-1)


def apply_training_blocks(model, performance, compile_fn):
    parameter = next(model.parameters())
    if parameter.dtype != torch.float32:
        raise PerformanceError("training_blocks requires FP32 model parameters and state; use "
                               "performance.precision for BF16 compute or set performance.compile=false")
    keys = set(model.state_dict())
    functions = {"advance": model._advance_reference, "write": model.anchors.write,
                 "nll_rows": nll_rows, "kl_rows": kl_rows}
    candidates = {}
    try:
        for name, function in functions.items():
            kwargs = {"fullgraph": True, "dynamic": False,
                      "options": {"emulate_precision_casts": True}}
            # Public, per-region options on newer torch builds. Never patch global
            # Dynamo settings, which could affect another model in this process.
            parameters = inspect.signature(compile_fn).parameters
            if "recompile_limit" in parameters:
                kwargs["recompile_limit"] = 64
            if "isolate_recompiles" in parameters:
                kwargs["isolate_recompiles"] = True
            compiled = compile_fn(function, **kwargs)
            if not callable(compiled):
                raise TypeError(f"compiler returned a non-callable for {name}")
            candidates[name] = _GuardedCallable(compiled, f"training_blocks.{name}")
        _preflight(model, performance, candidates)
    except PerformanceError:
        raise
    except Exception as error:
        raise PerformanceError(f"training_blocks compilation/preflight failed "
                               f"({type(error).__name__}: {error}); set performance.compile=false") from error
    if set(model.state_dict()) != keys:
        raise PerformanceError("training_blocks changed state_dict keys; set performance.compile=false")
    model.compiled_training = candidates
    return {"compile": True, "applied": True, "verified": True, "state": "applied",
            "targets": list(functions), "backend": "inductor", "dynamic": False,
            "recompile_limit": 64 if "recompile_limit" in parameters else "torch default",
            "isolated_cache": "isolate_recompiles" in parameters,
            "reason": "forward/backward preflight passed"}


def _preflight(model, performance, candidates):
    parameter = next(model.parameters())
    device, width = parameter.device, model.config.d_world
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" and torch.cuda.is_initialized() else None
    modes = {module: module.training for module in model.modules()}
    try:
        with torch.enable_grad(), autocast_context(performance, device):
            state = model.initial_state(2, device=device)
            delta = torch.full((2,), model.config.substep_seconds, device=device, dtype=torch.float64)
            advanced = candidates["advance"](state, delta, 1)
            tokens = torch.zeros(2, model.patches, width, device=device, requires_grad=True)
            coords = model.patch_coordinates.to(device)
            increment = candidates["write"](advanced.slots, tokens, coords)
            compute = tokens.to(torch.bfloat16) if performance.precision == "bfloat16" else tokens
            mu, logvar = compute * .1, compute * .2
            nll = candidates["nll_rows"](tokens.detach(), mu, logvar)
            kl = candidates["kl_rows"](mu, logvar, mu * .5, logvar * .5)
            loss = advanced.slots.square().mean() + increment.square().mean() + nll.mean() + kl.mean()
            gradients = torch.autograd.grad(loss, (tokens, *[p for p in model.parameters() if p.requires_grad]),
                                            allow_unused=True)
            finite = torch.isfinite(loss) & torch.stack([torch.isfinite(g).all() for g in gradients if g is not None]).all()
            if not bool(finite):
                raise PerformanceError("training_blocks preflight produced non-finite loss/gradients")
    finally:
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        for module, training in modes.items():
            module.training = training
