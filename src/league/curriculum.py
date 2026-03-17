"""Curriculum scheduler that manages board-size progression.

The curriculum advances through increasingly large boards as the
agent's win rate against built-in IPvGO bots improves.  Built-in
bots act as stable, noise-free progression gates so that the agent
cannot "cheat" by exploiting a weak self-play clone.

Example::

    curriculum = BoardCurriculum()
    # ... after training on 5x5 ...
    curriculum.update({"win_rate_vs_easy": 0.85})
    print(curriculum.current_size)  # 7 - automatically advanced
    print(curriculum.get_board_size())  # 5 or 7 (70/30 mix)
"""

from __future__ import annotations

import random
from typing import Any, ClassVar


class BoardCurriculum:
    """Curriculum scheduler that gates board-size advancement on win rate.

    The scheduler starts at the smallest board size (5x5) and
    advances to the next stage only when the agent achieves a
    sufficient win rate against the corresponding built-in IPvGO
    bot.  During a transition period a mixed sampling strategy
    returns the *current* size 70 % of the time and the *next* size
    30 % of the time, letting the agent begin adapting to the larger
    board before fully committing to it.

    Board sizes and their advancement criteria:

    * 5  -> 7  : ``win_rate_vs_easy``   >= 0.8
    * 7  -> 9  : ``win_rate_vs_medium`` >= 0.7
    * 9  -> 13 : ``win_rate_vs_medium`` >= 0.7
    * 13       : terminal stage

    Curriculum stages (used for opponent sampling):

    * ``"early"`` - boards 5 and 7
    * ``"mid"``   - board 9
    * ``"late"``  - board 13

    Args:
        transition_mix: Probability of sampling the *current* board
            size during transition (default 0.7 -> 70 %).
    """

    BOARD_SIZES: ClassVar[list[int]] = [5, 7, 9, 13]

    # (metric_key, threshold) required to advance from each size.
    _ADVANCE_THRESHOLDS: ClassVar[dict[int, tuple[str, float]]] = {
        5: ("win_rate_vs_easy", 0.8),
        7: ("win_rate_vs_medium", 0.7),
        9: ("win_rate_vs_medium", 0.7),
        13: ("win_rate_vs_hard", 1.1),  # unreachable - terminal
    }

    # Curriculum stage label for each board size.
    _STAGE_MAP: ClassVar[dict[int, str]] = {
        5: "early",
        7: "early",
        9: "mid",
        13: "late",
    }

    def __init__(self, transition_mix: float = 0.7) -> None:
        """Initialise the curriculum at the smallest board size.

        Args:
            transition_mix: Fraction of samples drawn from the
                *current* board size during transition (0 < mix <= 1).
        """
        if not (0.0 < transition_mix <= 1.0):
            raise ValueError(
                "transition_mix must be in (0, 1]; "
                f"got {transition_mix}"
            )
        self._stage_idx: int = 0
        self._metrics: dict[str, float] = {}
        self._transition_mix: float = transition_mix

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_size(self) -> int:
        """Current (primary) board size."""
        return self.BOARD_SIZES[self._stage_idx]

    @property
    def next_size(self) -> int | None:
        """Next board size, or ``None`` when at the terminal stage."""
        idx = self._stage_idx + 1
        if idx < len(self.BOARD_SIZES):
            return self.BOARD_SIZES[idx]
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_board_size(self) -> int:
        """Sample a board size for the next episode.

        Returns the *current* board size with probability
        ``transition_mix`` and the *next* board size with probability
        ``1 - transition_mix``.  When there is no next size (terminal
        stage) always returns the current size.

        Returns:
            Board side length (an element of :attr:`BOARD_SIZES`).
        """
        next_size = self.next_size
        if next_size is not None and random.random() > self._transition_mix:
            return next_size
        return self.current_size

    def update(self, metrics: dict[str, float]) -> None:
        """Ingest the latest evaluation metrics and advance if ready.

        Stores the metrics and calls :meth:`should_advance`.  If the
        threshold has been reached the curriculum moves to the next
        stage automatically.

        Args:
            metrics: Dict of evaluation metrics such as
                ``{"win_rate_vs_easy": 0.85, ...}``.
        """
        self._metrics = dict(metrics)
        if self.should_advance():
            self._stage_idx = min(
                self._stage_idx + 1, len(self.BOARD_SIZES) - 1
            )

    def should_advance(self) -> bool:
        """Return ``True`` if the win-rate threshold has been met.

        Returns:
            ``True`` when the required metric exceeds the advancement
            threshold for the current board size and there is a next
            stage to advance to; ``False`` otherwise.
        """
        if self._stage_idx >= len(self.BOARD_SIZES) - 1:
            return False
        size = self.current_size
        metric_key, threshold = self._ADVANCE_THRESHOLDS[size]
        return float(self._metrics.get(metric_key, 0.0)) >= threshold

    def get_stage(self) -> str:
        """Return the curriculum stage label for the current board size.

        Returns:
            One of ``"early"``, ``"mid"``, or ``"late"``.
        """
        return self._STAGE_MAP[self.current_size]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the curriculum state.

        Returns:
            Dict with keys ``"stage_idx"`` and ``"metrics"``.
        """
        return {
            "stage_idx": self._stage_idx,
            "metrics": dict(self._metrics),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore curriculum state from a snapshot.

        Args:
            state: Dict previously returned by :meth:`state_dict`.
        """
        self._stage_idx = int(state["stage_idx"])
        self._metrics = dict(state.get("metrics", {}))


__all__ = ["BoardCurriculum"]
