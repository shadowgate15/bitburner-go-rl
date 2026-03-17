"""Checkpoint manager for saving and organising model snapshots.

The :class:`CheckpointManager` handles three kinds of saves:

* **Latest** - ``checkpoints/latest.pt`` - overwritten every save.
* **Best**   - ``checkpoints/best.pt``   - only written when a new
  high-water metric is reached.
* **League** - ``checkpoints/league/step_<N>.pt`` - a permanent copy
  added to the opponent pool's historical archive.

The checkpoint format is compatible with the training loop in
:mod:`src.train.train` and with :meth:`OpponentPool.load_model_opponent`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """Save and organise training checkpoints.

    Manages three separate save locations:

    * ``<checkpoint_dir>/latest.pt`` - always overwritten.
    * ``<checkpoint_dir>/best.pt``   - saved when a tracked metric
      exceeds the current best value.
    * ``<checkpoint_dir>/league/step_<N>.pt`` - permanent snapshots
      added to the league opponent pool.

    Args:
        checkpoint_dir: Root directory for all checkpoints.  Created
            (along with the ``league/`` sub-directory) if absent.
        best_metric_key: Metric used to decide whether to overwrite
            ``best.pt`` (default ``"win_rate_vs_medium"``).
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        best_metric_key: str = "win_rate_vs_medium",
    ) -> None:
        """Initialise the checkpoint manager and create directories.

        Args:
            checkpoint_dir: Root directory for saved checkpoints.
            best_metric_key: Key in the metrics dict used to track the
                best-ever model.
        """
        self._dir = Path(checkpoint_dir)
        self._league_dir = self._dir / "league"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._league_dir.mkdir(parents=True, exist_ok=True)
        self._best_metric_key = best_metric_key
        self._best_value: float = float("-inf")

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        actor: Any,
        critic: Any,
        optimizer: Any,
        step: int,
        cfg: Any,
        curriculum_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble a checkpoint payload dict.

        Args:
            actor: TorchRL actor module.
            critic: TorchRL critic module.
            optimizer: PyTorch optimiser.
            step: Current training step.
            cfg: :class:`~src.train.train.TrainConfig` instance.
            curriculum_state: Optional dict from
                :meth:`~src.league.curriculum.BoardCurriculum.state_dict`.

        Returns:
            Dict suitable for ``torch.save``.
        """
        payload: dict[str, Any] = {
            "step": step,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": cfg,
        }
        if curriculum_state is not None:
            payload["curriculum_state"] = curriculum_state
        return payload

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_latest(
        self,
        actor: Any,
        critic: Any,
        optimizer: Any,
        step: int,
        cfg: Any,
        curriculum_state: dict[str, Any] | None = None,
    ) -> str:
        """Overwrite ``checkpoints/latest.pt``.

        Args:
            actor: Trained TorchRL actor module.
            critic: Trained TorchRL critic module.
            optimizer: PyTorch optimiser.
            step: Current training step / iteration count.
            cfg: :class:`~src.train.train.TrainConfig`.
            curriculum_state: Serialised curriculum state (optional).

        Returns:
            Absolute path of the saved file as a string.
        """
        path = self._dir / "latest.pt"
        torch.save(
            self._build_payload(
                actor, critic, optimizer, step, cfg, curriculum_state
            ),
            path,
        )
        return str(path)

    def maybe_save_best(
        self,
        metrics: dict[str, float],
        actor: Any,
        critic: Any,
        optimizer: Any,
        step: int,
        cfg: Any,
        curriculum_state: dict[str, Any] | None = None,
    ) -> bool:
        """Save ``checkpoints/best.pt`` if the tracked metric improved.

        Compares ``metrics[best_metric_key]`` against the stored
        high-water mark.  Saves and updates the mark when a new best
        is achieved.

        Args:
            metrics: Dict of evaluation metrics (e.g. from
                :func:`~src.league.evaluation.evaluate`).
            actor: Trained TorchRL actor module.
            critic: Trained TorchRL critic module.
            optimizer: PyTorch optimiser.
            step: Current training step.
            cfg: :class:`~src.train.train.TrainConfig`.
            curriculum_state: Serialised curriculum state (optional).

        Returns:
            ``True`` if a new best was saved, ``False`` otherwise.
        """
        value = float(metrics.get(self._best_metric_key, float("-inf")))
        if value <= self._best_value:
            return False
        self._best_value = value
        path = self._dir / "best.pt"
        torch.save(
            self._build_payload(
                actor, critic, optimizer, step, cfg, curriculum_state
            ),
            path,
        )
        return True

    def add_to_league(
        self,
        actor: Any,
        critic: Any,
        step: int,
        cfg: Any,
        curriculum_state: dict[str, Any] | None = None,
    ) -> str:
        """Save a permanent league snapshot.

        The file is written to
        ``checkpoints/league/step_<N>.pt``.  League snapshots are
        never overwritten; each step produces a unique file that can
        later be loaded by
        :meth:`~src.league.opponents.OpponentPool.load_model_opponent`.

        Args:
            actor: TorchRL actor module.
            critic: TorchRL critic module.
            step: Current training step (used in the filename).
            cfg: :class:`~src.train.train.TrainConfig`.
            curriculum_state: Serialised curriculum state (optional).

        Returns:
            Absolute path of the saved league snapshot.
        """
        path = self._league_dir / f"step_{step}.pt"
        payload: dict[str, Any] = {
            "step": step,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "cfg": cfg,
        }
        if curriculum_state is not None:
            payload["curriculum_state"] = curriculum_state
        torch.save(payload, path)
        return str(path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_value(self) -> float:
        """Current best metric value (``-inf`` before first save)."""
        return self._best_value

    @property
    def checkpoint_dir(self) -> Path:
        """Root checkpoint directory as a :class:`pathlib.Path`."""
        return self._dir

    @property
    def league_dir(self) -> Path:
        """League checkpoint sub-directory as a :class:`pathlib.Path`."""
        return self._league_dir


__all__ = ["CheckpointManager"]
