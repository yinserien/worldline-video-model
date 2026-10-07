"""A component's actual output must satisfy the contract before losses or PNGs."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from wpm_video.decoder import decode_latents, decoder_loss, render_latents


class _OutputDecoder(nn.Module):
    output_size = 32

    def __init__(self, output):
        super().__init__()
        self.output = output
        self.called_training = None

    def forward(self, latents):
        self.called_training = self.training
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class DecoderOutputContractTests(unittest.TestCase):
    def test_loss_refuses_broadcasting_and_non_float_outputs(self):
        targets = torch.zeros(2, 3, 32, 32)
        for output in (torch.zeros(1, 3, 32, 32), torch.zeros(2, 1, 32, 32),
                       torch.zeros(2, 3, 1, 1), torch.zeros_like(targets, dtype=torch.int64),
                       None):
            with self.subTest(output_type=type(output)), self.assertRaisesRegex(
                    ValueError, "decoder output"):
                decoder_loss(output, targets, 1.0, 0.1)

    def test_render_rejects_invalid_output_before_png_and_restores_mode(self):
        latents = torch.zeros(16, 32)
        outputs = [None, torch.zeros(1, 1, 32, 32), torch.zeros(1, 3, 64, 64),
                   torch.full((1, 3, 32, 32), float('nan')),
                   torch.full((1, 3, 32, 32), -0.1), torch.full((1, 3, 32, 32), 1.1)]
        with tempfile.TemporaryDirectory() as temporary:
            for index, output in enumerate(outputs):
                decoder = _OutputDecoder(output).train()
                out = Path(temporary) / str(index)
                with self.subTest(index=index), self.assertRaisesRegex(ValueError,
                                                                      "decoder output"):
                    render_latents(decoder, {1: (latents, {})}, out, torch.device('cpu'))
                self.assertTrue(decoder.training)
                self.assertFalse(decoder.called_training)
                self.assertEqual(list(out.glob('*.png')), [])

    def test_decode_restores_mode_when_a_custom_forward_fails(self):
        decoder = _OutputDecoder(RuntimeError('component failed')).train()
        with self.assertRaisesRegex(RuntimeError, 'component failed'):
            decode_latents(decoder, {1: torch.zeros(16, 32)}, torch.device('cpu'))
        self.assertTrue(decoder.training)
        self.assertFalse(decoder.called_training)
