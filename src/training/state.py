"""TrainState: the single source of truth for mutable run counters.

The trainer holds ``self.state``; every mutable run counter lives here.
``tokens_per_step`` is NOT state (it is a run constant, not mutated during
the run); the data loader is not state either (its memmap cursor is
checkpointed separately as ``data_state``).
"""

from dataclasses import dataclass


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    run_elapsed_seconds: float = 0.0
    wandb_run_id: str | None = None
