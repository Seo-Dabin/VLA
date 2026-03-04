"""Loss-convergence-based curriculum controller for 3-stage training.

Monitors validation loss over a patience window and transitions to the
next training stage when improvement falls below a threshold.

Stages:
  1. Depth prediction (depth decoder only)
  2. Image reconstruction (depth + image decoders)
  3. Visual token generation (all decoders)
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class CurriculumController:
    """Loss convergence-based stage transition controller.

    Monitors validation loss per stage and triggers stage transitions
    when the loss improvement over a patience window falls below a
    minimum threshold.

    Args:
        patience_epochs: Number of epochs to wait before checking convergence.
        min_improvement: Minimum relative improvement to continue current stage.
        max_epochs_per_stage: Hard cap on epochs per stage (fallback).
        total_stages: Number of curriculum stages.
    """

    def __init__(
        self,
        patience_epochs: int = 5,
        min_improvement: float = 0.01,
        max_epochs_per_stage: int = 30,
        total_stages: int = 3,
    ) -> None:
        self.patience_epochs = patience_epochs
        self.min_improvement = min_improvement
        self.max_epochs_per_stage = max_epochs_per_stage
        self.total_stages = total_stages

        self._current_stage = 1
        self._stage_epoch_count = 0
        self._val_loss_history: deque[float] = deque(maxlen=patience_epochs + 1)
        self._best_val_loss = float("inf")

    @property
    def current_stage(self) -> int:
        """Return the current active training stage (1-indexed)."""
        return self._current_stage

    @property
    def active_stages(self) -> List[int]:
        """Return list of all active stages (cumulative).

        Stage 1: [1]
        Stage 2: [1, 2]
        Stage 3: [1, 2, 3]

        Returns:
            List of active stage numbers.
        """
        return list(range(1, self._current_stage + 1))

    @property
    def stage_epoch_count(self) -> int:
        """Number of epochs spent in the current stage."""
        return self._stage_epoch_count

    def should_advance(self) -> bool:
        """Check if the current stage should advance to next.

        Transitions when:
          - Patience window is full AND relative improvement < threshold
          - OR max_epochs_per_stage is reached

        Returns:
            True if stage should advance.
        """
        if self._current_stage >= self.total_stages:
            return False

        # Hard cap
        if self._stage_epoch_count >= self.max_epochs_per_stage:
            return True

        # Need enough history for patience check
        if len(self._val_loss_history) <= self.patience_epochs:
            return False

        # Check relative improvement over patience window
        recent_losses = list(self._val_loss_history)
        old_loss = recent_losses[0]
        new_loss = recent_losses[-1]

        if old_loss == 0:
            return False

        relative_improvement = (old_loss - new_loss) / abs(old_loss)
        return relative_improvement < self.min_improvement

    def update(self, val_loss: float) -> Tuple[bool, int]:
        """Update controller with new validation loss.

        Args:
            val_loss: Current epoch's validation loss for the primary active stage.

        Returns:
            advanced: Whether a stage transition occurred.
            new_stage: Current stage number after update.
        """
        self._stage_epoch_count += 1
        self._val_loss_history.append(val_loss)

        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss

        advanced = False
        if self.should_advance():
            self._advance_stage()
            advanced = True

        return advanced, self._current_stage

    def _advance_stage(self) -> None:
        """Advance to the next training stage."""
        old_stage = self._current_stage
        self._current_stage = min(self._current_stage + 1, self.total_stages)
        self._stage_epoch_count = 0
        self._val_loss_history.clear()
        self._best_val_loss = float("inf")
        print(f"[Curriculum] Stage transition: {old_stage} -> {self._current_stage}")

    def force_stage(self, stage: int) -> None:
        """Force a specific stage (for resuming from checkpoint).

        Args:
            stage: Target stage number (1-indexed).
        """
        self._current_stage = max(1, min(stage, self.total_stages))
        self._stage_epoch_count = 0
        self._val_loss_history.clear()
        self._best_val_loss = float("inf")

    def state_dict(self) -> dict:
        """Serialize controller state for checkpointing.

        Returns:
            Dictionary with controller state.
        """
        return {
            "current_stage": self._current_stage,
            "stage_epoch_count": self._stage_epoch_count,
            "val_loss_history": list(self._val_loss_history),
            "best_val_loss": self._best_val_loss,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore controller state from checkpoint.

        Args:
            state: Dictionary with saved controller state.
        """
        self._current_stage = state["current_stage"]
        self._stage_epoch_count = state["stage_epoch_count"]
        self._val_loss_history = deque(state["val_loss_history"], maxlen=self.patience_epochs + 1)
        self._best_val_loss = state["best_val_loss"]
