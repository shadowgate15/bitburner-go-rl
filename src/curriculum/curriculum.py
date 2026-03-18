"""Curriculum learning scheduler for the BitBurner IPvGO RL agent.

The :class:`GoCurriculumManager` dynamically adjusts the training
opponent and board size based on the agent's win rate, making the
curriculum progressively harder as the agent improves and easier when
it struggles.

Opponents (easiest → hardest):

1. ``"Netburners"``   - Easy
2. ``"Slum Snakes"``  - Spread
3. ``"Tetrads"``      - Martial
4. ``"The Black Hand"`` - Aggro
5. ``"Daedalus"``     - Mid
6. ``"Illuminati"``   - Hard

Board sizes (smallest → largest):

``[5, 7, 9, 13]``

Progression rules:

* win_rate > ``win_rate_threshold_up``  → advance difficulty
  (first increase opponent; if already at hardest, increase board size)
* win_rate < ``win_rate_threshold_down`` → retreat difficulty
  (first reduce opponent; if already at easiest, reduce board size)
* otherwise                             → no change

Stability features:

* Requires at least ``min_evaluations`` total evaluations before
  any level change.
* A ``cooldown_evals``-evaluation cooldown prevents rapid oscillation.
* Win rate is smoothed with a ``smoothing_window``-evaluation moving
  average before comparison against the thresholds.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Difficulty tables (ordered easiest → hardest)
# ---------------------------------------------------------------------------

#: Built-in BitBurner opponents ordered by increasing difficulty.
OPPONENTS: list[str] = [
    "Netburners",  # Easy
    "Slum Snakes",  # Spread
    "Tetrads",  # Martial
    "The Black Hand",  # Aggro
    "Daedalus",  # Mid
    "Illuminati",  # Hard
]

#: Board sizes ordered by increasing difficulty.
BOARD_SIZES: list[int] = [5, 7, 9, 13]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class CurriculumConfig:
    """Hyper-parameters for the curriculum scheduler.

    Attributes:
        win_rate_threshold_up: Advance difficulty when the smoothed
            win rate exceeds this value (default 0.7).
        win_rate_threshold_down: Retreat difficulty when the smoothed
            win rate falls below this value (default 0.3).
        min_evaluations: Minimum number of evaluation phases that must
            have completed before any level change is allowed.
        smoothing_window: Number of recent win-rate values used for
            the moving-average computation.
        cooldown_evals: Number of evaluation phases that must elapse
            after a level change before the next change is allowed.
            This prevents rapid oscillation between levels.
    """

    win_rate_threshold_up: float = 0.7
    win_rate_threshold_down: float = 0.3
    min_evaluations: int = 3
    smoothing_window: int = 5
    cooldown_evals: int = 2

    # Internal: win-rate history kept for the moving average.
    _win_rate_history: deque[float] = field(
        default_factory=deque, repr=False, compare=False
    )


# ---------------------------------------------------------------------------
# Curriculum manager
# ---------------------------------------------------------------------------


class GoCurriculumManager:
    """Manages curriculum difficulty for the Go RL agent.

    Tracks the current opponent and board size, exposes them via
    :meth:`get_current_config`, and updates them based on evaluation
    metrics supplied to :meth:`update`.

    Example::

        curriculum = GoCurriculumManager()
        env.reset(**curriculum.get_current_config())
        # … run training …
        curriculum.update({"win_rate": 0.8, "avg_reward": 1.2})
        env.reset(**curriculum.get_current_config())

    Args:
        config: Curriculum configuration.  Defaults to
            :class:`CurriculumConfig` with all default values.
        initial_opponent_idx: Index into :data:`OPPONENTS` to start
            from (0 = easiest).
        initial_board_size_idx: Index into :data:`BOARD_SIZES` to
            start from (0 = smallest).
    """

    def __init__(
        self,
        config: CurriculumConfig | None = None,
        initial_opponent_idx: int = 0,
        initial_board_size_idx: int = 0,
    ) -> None:
        """Initialise the curriculum manager.

        Args:
            config: Curriculum configuration.
            initial_opponent_idx: Starting opponent index.
            initial_board_size_idx: Starting board-size index.
        """
        if initial_opponent_idx not in range(len(OPPONENTS)):
            raise ValueError(
                f"initial_opponent_idx must be in "
                f"[0, {len(OPPONENTS) - 1}], "
                f"got {initial_opponent_idx}"
            )
        if initial_board_size_idx not in range(len(BOARD_SIZES)):
            raise ValueError(
                f"initial_board_size_idx must be in "
                f"[0, {len(BOARD_SIZES) - 1}], "
                f"got {initial_board_size_idx}"
            )

        self._cfg = config or CurriculumConfig()
        self._opponent_idx: int = initial_opponent_idx
        self._board_size_idx: int = initial_board_size_idx

        # Smoothed win-rate history (bounded by smoothing_window).
        self._win_rate_history: deque[float] = deque(
            maxlen=self._cfg.smoothing_window
        )

        # Total evaluation count since initialisation.
        self._eval_count: int = 0

        # Evaluations elapsed since the last level change (cooldown).
        # Initialised to cfg.cooldown_evals so the first change is
        # permitted as soon as min_evaluations evaluations have passed.
        self._evals_since_change: int = self._cfg.cooldown_evals

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def opponent_idx(self) -> int:
        """Current opponent index within :data:`OPPONENTS`."""
        return self._opponent_idx

    @property
    def board_size_idx(self) -> int:
        """Current board-size index within :data:`BOARD_SIZES`."""
        return self._board_size_idx

    @property
    def current_opponent(self) -> str:
        """Name of the current opponent."""
        return OPPONENTS[self._opponent_idx]

    @property
    def current_board_size(self) -> int:
        """Side length of the current board."""
        return BOARD_SIZES[self._board_size_idx]

    @property
    def eval_count(self) -> int:
        """Total number of evaluation phases completed so far."""
        return self._eval_count

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_current_config(self) -> dict[str, int | str]:
        """Return the current curriculum settings for env.reset().

        Returns:
            Dict with keys:

            * ``"opponent"`` - name of the current opponent (str).
            * ``"board_size"`` - side length of the current board (int).
        """
        return {
            "opponent": self.current_opponent,
            "board_size": self.current_board_size,
        }

    def update(self, metrics: dict[str, float]) -> None:
        """Update the curriculum based on evaluation metrics.

        Called once per evaluation phase.  Updates the internal win-rate
        history, checks stability constraints, and advances or retreats
        the difficulty level if the smoothed win rate crosses a
        threshold.

        Args:
            metrics: Dictionary that **must** contain ``"win_rate"``
                (float in ``[0, 1]``) and may optionally contain
                ``"avg_reward"`` (float) and ``"game_length"``
                (float or int).

        Raises:
            KeyError: If ``"win_rate"`` is missing from *metrics*.
        """
        win_rate: float = float(metrics["win_rate"])
        avg_reward: float = float(metrics.get("avg_reward", float("nan")))

        self._eval_count += 1
        self._evals_since_change += 1
        self._win_rate_history.append(win_rate)

        smoothed = self._smoothed_win_rate()

        logger.info(
            "[Curriculum] eval=%d | opponent=%s | board_size=%d | "
            "win_rate=%.3f | smoothed_win_rate=%.3f | "
            "avg_reward=%.4f",
            self._eval_count,
            self.current_opponent,
            self.current_board_size,
            win_rate,
            smoothed,
            avg_reward,
        )

        # ---- Stability gates ----
        if self._eval_count < self._cfg.min_evaluations:
            logger.debug(
                "[Curriculum] Skipping level check: only %d/%d "
                "evaluations completed.",
                self._eval_count,
                self._cfg.min_evaluations,
            )
            return

        if self._evals_since_change < self._cfg.cooldown_evals:
            logger.debug(
                "[Curriculum] Skipping level check: cooldown "
                "(%d/%d evaluations elapsed).",
                self._evals_since_change,
                self._cfg.cooldown_evals,
            )
            return

        # ---- Progression decision ----
        if smoothed > self._cfg.win_rate_threshold_up:
            self._advance_level()
        elif smoothed < self._cfg.win_rate_threshold_down:
            self._retreat_level()
        else:
            logger.debug(
                "[Curriculum] No change (smoothed win_rate %.3f "
                "in [%.1f, %.1f]).",
                smoothed,
                self._cfg.win_rate_threshold_down,
                self._cfg.win_rate_threshold_up,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _smoothed_win_rate(self) -> float:
        """Compute the moving-average win rate over the history window.

        Returns:
            Mean of all values currently in the history deque, or
            ``0.0`` if the history is empty.
        """
        if not self._win_rate_history:
            return 0.0
        return sum(self._win_rate_history) / len(self._win_rate_history)

    def _advance_level(self) -> None:
        """Increase difficulty by one step.

        Advances the opponent index first; only increases the board size
        when the opponent is already at the hardest level.  Logs the
        change and resets the cooldown counter.
        """
        if self._opponent_idx < len(OPPONENTS) - 1:
            old_opponent = self.current_opponent
            self._opponent_idx += 1
            logger.info(
                "[Curriculum] ADVANCE opponent: %s → %s "
                "(board_size=%d unchanged)",
                old_opponent,
                self.current_opponent,
                self.current_board_size,
            )
            self._evals_since_change = 0
        elif self._board_size_idx < len(BOARD_SIZES) - 1:
            old_size = self.current_board_size
            self._board_size_idx += 1
            logger.info(
                "[Curriculum] ADVANCE board_size: %d → %d "
                "(opponent=%s unchanged)",
                old_size,
                self.current_board_size,
                self.current_opponent,
            )
            self._evals_since_change = 0
        else:
            logger.info(
                "[Curriculum] Already at maximum difficulty "
                "(opponent=%s, board_size=%d).",
                self.current_opponent,
                self.current_board_size,
            )

    def _retreat_level(self) -> None:
        """Decrease difficulty by one step.

        Reduces the opponent index first; only reduces the board size
        when the opponent is already at the easiest level.  Logs the
        change and resets the cooldown counter.
        """
        if self._opponent_idx > 0:
            old_opponent = self.current_opponent
            self._opponent_idx -= 1
            logger.info(
                "[Curriculum] RETREAT opponent: %s → %s "
                "(board_size=%d unchanged)",
                old_opponent,
                self.current_opponent,
                self.current_board_size,
            )
            self._evals_since_change = 0
        elif self._board_size_idx > 0:
            old_size = self.current_board_size
            self._board_size_idx -= 1
            logger.info(
                "[Curriculum] RETREAT board_size: %d → %d "
                "(opponent=%s unchanged)",
                old_size,
                self.current_board_size,
                self.current_opponent,
            )
            self._evals_since_change = 0
        else:
            logger.info(
                "[Curriculum] Already at minimum difficulty "
                "(opponent=%s, board_size=%d).",
                self.current_opponent,
                self.current_board_size,
            )


__all__ = [
    "BOARD_SIZES",
    "OPPONENTS",
    "CurriculumConfig",
    "GoCurriculumManager",
]
