"""Production scope integration, atomic setup, backward preflight and resume guards."""
import copy
import unittest
from unittest.mock import patch

import torch

from wpm_video import ModelConfig, PerformanceConfig, RunConfig, VideoWorldModel
from wpm_video.performance import apply_compile, PerformanceError, require_matching_policy, training_policy


def model(dtype=torch.float32):
    config = ModelConfig(d_world=32, d_hidden=64, slots=4, heads=4)
    return VideoWorldModel(config, 32, 16, .5).to(dtype=dtype)


class TrainingBlockTests(unittest.TestCase):
    def test_config_and_reference_defaults(self):
        self.assertEqual(PerformanceConfig().compile_scope, "predictor")
        self.assertFalse(PerformanceConfig().compile_fallback)
        legacy = PerformanceConfig("float32", "reference", False, False, True, True)
        self.assertTrue(legacy.pin_memory)
        self.assertTrue(legacy.non_blocking)
        self.assertEqual(legacy.compile_scope, "predictor")
        for kwargs in ({"compile_scope": "all"}, {"compile_fallback": 1}):
            with self.assertRaises(ValueError):
                RunConfig(performance=PerformanceConfig(**kwargs)).validate()

    def test_setup_preserves_rng_modes_gradients_and_checkpoint_keys(self):
        value = model()
        value.acceleration[0].eval()
        value.slot_init.grad = torch.ones_like(value.slot_init)
        modes = [m.training for m in value.modules()]
        keys = set(value.state_dict())
        rng = torch.get_rng_state().clone()
        calls = []
        def compiler(function, **kwargs):
            calls.append(kwargs)
            return function
        status = apply_compile(value, PerformanceConfig(compile=True, compile_scope="training_blocks"), compiler)
        self.assertTrue(status["verified"])
        self.assertEqual(set(status["targets"]), {"advance", "write", "nll_rows", "kl_rows"})
        self.assertEqual(set(value.state_dict()), keys)
        self.assertEqual([m.training for m in value.modules()], modes)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(value.slot_init.grad, torch.ones_like(value.slot_init)))
        self.assertTrue(all(call["options"]["emulate_precision_casts"] for call in calls))

    def test_installed_hooks_match_reference_loss_and_gradients(self):
        from wpm_video.train import forward_window
        base = model()
        optimized = copy.deepcopy(base)
        config = RunConfig(model=base.config)
        config.data.context_chunks = 2
        batch = {"observed": torch.randn(2, 2, 16, 32),
                 "end_seconds": torch.tensor([[.5, 1.], [.4, .9]], dtype=torch.float64),
                 "context_substeps": [2, 2], "targets": {
                     (1, 1): {"tokens": torch.randn(2, 16, 32),
                              "delta_seconds": torch.tensor([.5, .4], dtype=torch.float64),
                              "index": torch.tensor([0]), "valid": torch.tensor([True, False]), "substeps": 2}}}
        apply_compile(optimized, PerformanceConfig(compile=True, compile_scope="training_blocks"),
                      lambda function, **kwargs: function)
        for value in (base, optimized):
            torch.manual_seed(7)
            loss, _ = forward_window(value, batch, config, stats_mode="tensor")
            loss.backward()
            if value is base:
                reference = loss.detach()
                gradients = {n: p.grad.clone() for n, p in value.named_parameters() if p.grad is not None}
            else:
                torch.testing.assert_close(loss, reference)
                for n, p in value.named_parameters():
                    if p.grad is not None:torch.testing.assert_close(p.grad, gradients[n])

    def test_failed_backward_never_attaches_partial_targets(self):
        class BadBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor):return tensor.clone()
            @staticmethod
            def backward(ctx, grad):raise RuntimeError("inductor backward failed")
        value = model()
        def compiler(function, **kwargs):
            if function.__name__ == "nll_rows":
                return lambda *args: BadBackward.apply(function(*args))
            return function
        with self.assertRaises(PerformanceError):
            apply_compile(value, PerformanceConfig(compile=True, compile_scope="training_blocks"), compiler)
        self.assertIsNone(value.compiled_training)
        self.assertIsNone(value.compiled_predictor)

    def test_explicit_setup_fallback_is_recorded_as_reference(self):
        value = model(torch.float64)
        performance = PerformanceConfig(compile=True, compile_scope="training_blocks", compile_fallback=True)
        with self.assertWarnsRegex(RuntimeWarning, "reference"):
            status = apply_compile(value, performance)
        self.assertEqual(status["state"], "fallback")
        self.assertEqual(status["scope"], "none")
        self.assertFalse(training_policy(performance, value, "cpu", status)["compile"])
        self.assertIsNone(value.compiled_training)
        apply_compile(value, PerformanceConfig())
        self.assertIsNone(value.compiled_training)

    def test_scope_changes_and_runtime_failures_cannot_silently_resume(self):
        old = {"compile": True}
        require_matching_policy(old, {"compile": True, "compile_scope": "predictor"}, "performance")
        with self.assertRaisesRegex(PerformanceError, "compile_scope"):
            require_matching_policy(old, {"compile": True, "compile_scope": "training_blocks"}, "performance")
        require_matching_policy({"compile": False}, {"compile": False, "compile_scope": "none"}, "performance")
        value = model()
        performance = PerformanceConfig(compile=True, compile_scope="training_blocks", compile_fallback=True)
        apply_compile(value, performance, lambda function, **kwargs: function)
        from wpm_video.performance import _GuardedCallable
        def broken(*args):raise RuntimeError("BackendCompilerFailed inductor")
        value.compiled_training["advance"] = _GuardedCallable(broken, "advance")
        with self.assertRaises(PerformanceError):
            value.advance(value.initial_state(2), torch.tensor([.2, .3]), substeps=2)
        self.assertIsNotNone(value.compiled_training)


if __name__ == "__main__":unittest.main()
