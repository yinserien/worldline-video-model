"""Persistent world state: content-addressed slots plus an explicit rate of change.

The state is deliberately not a clip embedding. It carries

- ``slots``: a fixed set of content-addressed latent slots addressed by similarity,
- ``velocity``: the explicit rate of change of those slots, so the model can keep
  moving without any observation,
- ``time`` and ``step``: where the state is on the real timeline.

Everything that changes the state must go through the innovation writer in
``model.py``; this module only stores, transports and resets it.
"""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class WorldState:
    slots: torch.Tensor      # (B, M, D)
    velocity: torch.Tensor   # (B, M, D)
    time: torch.Tensor       # (B,) seconds on the source timeline
    step: torch.Tensor       # (B,) chunks observed so far

    @staticmethod
    def init(batch_size: int, slots: int, d_world: int, device=None, dtype=torch.float32) -> "WorldState":
        return WorldState(
            slots=torch.zeros(batch_size, slots, d_world, device=device, dtype=dtype),
            velocity=torch.zeros(batch_size, slots, d_world, device=device, dtype=dtype),
            # Timestamps are bookkeeping, not activations: they stay float64 so a
            # chunk boundary never rounds onto the state time and turns a real
            # observation into a zero-length step.
            time=torch.zeros(batch_size, device=device, dtype=torch.float64),
            step=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def clone(self) -> "WorldState":
        return WorldState(self.slots.clone(), self.velocity.clone(), self.time.clone(), self.step.clone())

    def detach(self) -> "WorldState":
        return WorldState(self.slots.detach(), self.velocity.detach(), self.time.detach(), self.step.detach())

    def to(self, device=None, dtype=None) -> "WorldState":
        return WorldState(
            self.slots.to(device=device, dtype=dtype),
            self.velocity.to(device=device, dtype=dtype),
            self.time.to(device=device, dtype=torch.float64),
            self.step.to(device=device),
        )

    def select(self, index) -> "WorldState":
        return WorldState(self.slots[index], self.velocity[index], self.time[index], self.step[index])

    def __getitem__(self, index) -> "WorldState":
        return self.select(index)

    def reset(self, mask: torch.Tensor, initial_slots: torch.Tensor | None = None) -> "WorldState":
        """Reset selected rows to the initial state (new independent video).

        ``initial_slots`` must be the learned initial state the model starts a fresh
        video with; the zero tensor is only a fallback for callers that have none,
        and ``VideoWorldModel.reset_state`` always passes the learned values so a
        reset row is identical to a freshly created one.
        """
        keep = (~mask).to(self.slots.dtype).view(-1, 1, 1)
        initial = torch.zeros_like(self.slots) if initial_slots is None else initial_slots
        return WorldState(
            self.slots * keep + initial * (1 - keep),
            self.velocity * keep,
            self.time * keep.view(-1),
            (self.step * keep.view(-1).long()),
        )

    def save(self, path) -> None:
        path = Path(path)  # accepts str or Path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({name: getattr(self, name).detach().cpu() for name in
                    ("slots", "velocity", "time", "step")}, path)

    @staticmethod
    def load(path, device=None, dtype=None) -> "WorldState":
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return WorldState(payload["slots"], payload["velocity"], payload["time"], payload["step"]).to(
            device=device, dtype=dtype
        )

    def batch_size(self) -> int:
        return self.slots.shape[0]

    def advance_clock(self, delta_seconds: float) -> "WorldState":
        return WorldState(self.slots, self.velocity, self.time + delta_seconds, self.step)
