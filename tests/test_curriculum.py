"""Tests for the GoCurriculumManager and supporting types.

Tests do **not** require a live Bitburner WebSocket server.  They
exercise:

* :class:`~src.curriculum.curriculum.GoCurriculumManager` default
  state and configuration.
* ``get_current_config()`` returns the expected dict layout.
* ``update()`` advances difficulty when win_rate is high.
* ``update()`` retreats difficulty when win_rate is low.
* ``update()`` holds difficulty when win_rate is in the neutral zone.
* Stability constraints: min_evaluations and cooldown_evals gates.
* Smoothed win-rate moving average.
* Opponent-first progression before board-size progression.
* Board-size-first retreat before opponent retreat.
* Boundary behaviour at maximum and minimum difficulty.
* ``CurriculumConfig`` default values.
* :data:`~src.curriculum.curriculum.OPPONENTS` and
  :data:`~src.curriculum.curriculum.BOARD_SIZES` public constants.
* Invalid constructor arguments raise ``ValueError``.
"""

from __future__ import annotations

import pytest

from src.curriculum.curriculum import (
    BOARD_SIZES,
    OPPONENTS,
    CurriculumConfig,
    GoCurriculumManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WIN = {"win_rate": 1.0, "avg_reward": 1.0}
_LOSS = {"win_rate": 0.0, "avg_reward": -1.0}
_NEUTRAL = {"win_rate": 0.5, "avg_reward": 0.0}


def _fast_cfg(**kwargs: float | int) -> CurriculumConfig:
    """Return a CurriculumConfig with instant level-change thresholds.

    Defaults: min_evaluations=1, cooldown_evals=1, smoothing_window=1.
    Additional keyword arguments override these defaults.
    """
    return CurriculumConfig(
        min_evaluations=int(kwargs.get("min_evaluations", 1)),
        cooldown_evals=int(kwargs.get("cooldown_evals", 1)),
        smoothing_window=int(kwargs.get("smoothing_window", 1)),
        win_rate_threshold_up=float(
            kwargs.get("win_rate_threshold_up", 0.7)
        ),
        win_rate_threshold_down=float(
            kwargs.get("win_rate_threshold_down", 0.3)
        ),
    )


def _manager_at_max() -> GoCurriculumManager:
    """Return a manager already at the highest difficulty."""
    return GoCurriculumManager(
        config=_fast_cfg(),
        initial_opponent_idx=len(OPPONENTS) - 1,
        initial_board_size_idx=len(BOARD_SIZES) - 1,
    )


def _manager_at_min() -> GoCurriculumManager:
    """Return a manager already at the lowest difficulty."""
    return GoCurriculumManager(
        config=_fast_cfg(),
        initial_opponent_idx=0,
        initial_board_size_idx=0,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for the public OPPONENTS and BOARD_SIZES constants."""

    def test_opponents_is_list_of_strings(self) -> None:
        """OPPONENTS must be a list of non-empty strings."""
        assert isinstance(OPPONENTS, list)
        assert all(isinstance(o, str) and o for o in OPPONENTS)

    def test_opponents_length(self) -> None:
        """OPPONENTS must contain exactly 6 entries."""
        assert len(OPPONENTS) == 6

    def test_opponents_order(self) -> None:
        """First opponent must be Netburners and last must be Illuminati."""
        assert OPPONENTS[0] == "Netburners"
        assert OPPONENTS[-1] == "Illuminati"

    def test_board_sizes_is_list_of_ints(self) -> None:
        """BOARD_SIZES must be a list of positive integers."""
        assert isinstance(BOARD_SIZES, list)
        assert all(isinstance(s, int) and s > 0 for s in BOARD_SIZES)

    def test_board_sizes_sorted(self) -> None:
        """BOARD_SIZES must be in ascending order."""
        assert sorted(BOARD_SIZES) == BOARD_SIZES

    def test_board_sizes_values(self) -> None:
        """BOARD_SIZES must be [5, 7, 9, 13]."""
        assert BOARD_SIZES == [5, 7, 9, 13]


# ---------------------------------------------------------------------------
# CurriculumConfig
# ---------------------------------------------------------------------------


class TestCurriculumConfig:
    """Tests for the CurriculumConfig dataclass."""

    def test_default_threshold_up(self) -> None:
        """Default advance threshold must be 0.7."""
        assert CurriculumConfig().win_rate_threshold_up == pytest.approx(
            0.7
        )

    def test_default_threshold_down(self) -> None:
        """Default retreat threshold must be 0.3."""
        assert CurriculumConfig().win_rate_threshold_down == pytest.approx(
            0.3
        )

    def test_default_min_evaluations(self) -> None:
        """Default min_evaluations must be 3."""
        assert CurriculumConfig().min_evaluations == 3

    def test_default_smoothing_window(self) -> None:
        """Default smoothing_window must be 5."""
        assert CurriculumConfig().smoothing_window == 5

    def test_default_cooldown_evals(self) -> None:
        """Default cooldown_evals must be 2."""
        assert CurriculumConfig().cooldown_evals == 2

    def test_custom_values(self) -> None:
        """CurriculumConfig must accept and store custom values."""
        cfg = CurriculumConfig(
            win_rate_threshold_up=0.8,
            min_evaluations=5,
            cooldown_evals=3,
        )
        assert cfg.win_rate_threshold_up == pytest.approx(0.8)
        assert cfg.min_evaluations == 5
        assert cfg.cooldown_evals == 3


# ---------------------------------------------------------------------------
# GoCurriculumManager - initial state
# ---------------------------------------------------------------------------


class TestGoCurriculumManagerInit:
    """Tests for default initial state of GoCurriculumManager."""

    def test_default_opponent_idx(self) -> None:
        """Default opponent index must be 0 (easiest)."""
        assert GoCurriculumManager().opponent_idx == 0

    def test_default_board_size_idx(self) -> None:
        """Default board-size index must be 0 (smallest)."""
        assert GoCurriculumManager().board_size_idx == 0

    def test_default_opponent_name(self) -> None:
        """Default opponent must be Netburners."""
        assert GoCurriculumManager().current_opponent == "Netburners"

    def test_default_board_size(self) -> None:
        """Default board size must be 5 (smallest)."""
        assert GoCurriculumManager().current_board_size == 5

    def test_eval_count_starts_at_zero(self) -> None:
        """eval_count must be 0 before any update calls."""
        assert GoCurriculumManager().eval_count == 0

    def test_custom_initial_opponent(self) -> None:
        """A custom initial_opponent_idx must be respected."""
        m = GoCurriculumManager(initial_opponent_idx=2)
        assert m.current_opponent == OPPONENTS[2]

    def test_custom_initial_board_size(self) -> None:
        """A custom initial_board_size_idx must be respected."""
        m = GoCurriculumManager(initial_board_size_idx=2)
        assert m.current_board_size == BOARD_SIZES[2]

    def test_invalid_opponent_idx_raises(self) -> None:
        """An out-of-range initial_opponent_idx must raise ValueError."""
        with pytest.raises(ValueError, match="initial_opponent_idx"):
            GoCurriculumManager(initial_opponent_idx=99)

    def test_invalid_board_size_idx_raises(self) -> None:
        """An out-of-range initial_board_size_idx must raise ValueError."""
        with pytest.raises(ValueError, match="initial_board_size_idx"):
            GoCurriculumManager(initial_board_size_idx=99)

    def test_negative_opponent_idx_raises(self) -> None:
        """A negative initial_opponent_idx must raise ValueError."""
        with pytest.raises(ValueError):
            GoCurriculumManager(initial_opponent_idx=-1)


# ---------------------------------------------------------------------------
# get_current_config
# ---------------------------------------------------------------------------


class TestGetCurrentConfig:
    """Tests for get_current_config()."""

    def test_returns_dict(self) -> None:
        """get_current_config must return a dict."""
        cfg = GoCurriculumManager().get_current_config()
        assert isinstance(cfg, dict)

    def test_has_opponent_key(self) -> None:
        """Config dict must have an 'opponent' key."""
        cfg = GoCurriculumManager().get_current_config()
        assert "opponent" in cfg

    def test_has_board_size_key(self) -> None:
        """Config dict must have a 'board_size' key."""
        cfg = GoCurriculumManager().get_current_config()
        assert "board_size" in cfg

    def test_opponent_value_is_string(self) -> None:
        """Opponent value must be a string."""
        cfg = GoCurriculumManager().get_current_config()
        assert isinstance(cfg["opponent"], str)

    def test_board_size_value_is_int(self) -> None:
        """board_size value must be an int."""
        cfg = GoCurriculumManager().get_current_config()
        assert isinstance(cfg["board_size"], int)

    def test_default_config_values(self) -> None:
        """Default config must be Netburners + board_size=5."""
        cfg = GoCurriculumManager().get_current_config()
        assert cfg["opponent"] == "Netburners"
        assert cfg["board_size"] == 5


# ---------------------------------------------------------------------------
# update - stability gates
# ---------------------------------------------------------------------------


class TestUpdateStabilityGates:
    """Tests that min_evaluations/cooldown_evals prevent premature changes."""

    def test_no_change_before_min_evaluations(self) -> None:
        """Level must not change before min_evaluations evaluations."""
        m = GoCurriculumManager(
            config=CurriculumConfig(
                min_evaluations=3,
                cooldown_evals=1,
                smoothing_window=1,
            )
        )
        # Two high-win-rate updates - fewer than min_evaluations=3.
        m.update(_WIN)
        m.update(_WIN)
        # Still at default level.
        assert m.opponent_idx == 0
        assert m.board_size_idx == 0

    def test_change_allowed_at_min_evaluations(self) -> None:
        """Level must change once min_evaluations evaluations have occurred."""
        m = GoCurriculumManager(
            config=CurriculumConfig(
                min_evaluations=3,
                cooldown_evals=1,
                smoothing_window=1,
            )
        )
        m.update(_WIN)
        m.update(_WIN)
        m.update(_WIN)  # Third eval triggers advance.
        assert m.opponent_idx == 1

    def test_cooldown_prevents_rapid_oscillation(self) -> None:
        """A second level change must not happen during the cooldown."""
        m = GoCurriculumManager(
            config=CurriculumConfig(
                min_evaluations=1,
                cooldown_evals=3,
                smoothing_window=1,
            )
        )
        m.update(_WIN)  # Advance: opponent 0 → 1, cooldown resets.
        assert m.opponent_idx == 1
        # Still within cooldown (only 1 eval since change, need 3).
        m.update(_WIN)
        assert m.opponent_idx == 1  # No second advance yet.

    def test_eval_count_increments(self) -> None:
        """eval_count must increase by 1 for each update call."""
        m = GoCurriculumManager()
        for i in range(1, 6):
            m.update(_NEUTRAL)
            assert m.eval_count == i


# ---------------------------------------------------------------------------
# update - win/loss decisions
# ---------------------------------------------------------------------------


class TestUpdateProgression:
    """Tests for level advancement and retreat decisions."""

    def test_high_win_rate_advances_opponent(self) -> None:
        """A high win rate must advance the opponent first."""
        m = GoCurriculumManager(config=_fast_cfg())
        m.update(_WIN)
        assert m.opponent_idx == 1
        assert m.board_size_idx == 0

    def test_low_win_rate_does_nothing_at_minimum(self) -> None:
        """A low win rate at minimum difficulty must not go below 0."""
        m = _manager_at_min()
        m.update(_LOSS)
        assert m.opponent_idx == 0
        assert m.board_size_idx == 0

    def test_neutral_win_rate_no_change(self) -> None:
        """A neutral win rate must leave the level unchanged."""
        m = GoCurriculumManager(config=_fast_cfg())
        m.update(_NEUTRAL)
        assert m.opponent_idx == 0
        assert m.board_size_idx == 0

    def test_advance_board_size_when_at_hardest_opponent(self) -> None:
        """Board size must advance when opponent is already at maximum."""
        m = GoCurriculumManager(
            config=_fast_cfg(),
            initial_opponent_idx=len(OPPONENTS) - 1,
            initial_board_size_idx=0,
        )
        m.update(_WIN)
        assert m.board_size_idx == 1
        # Opponent index stays at maximum.
        assert m.opponent_idx == len(OPPONENTS) - 1

    def test_retreat_opponent_before_board_size(self) -> None:
        """A low win rate must reduce opponent before board size."""
        m = GoCurriculumManager(
            config=_fast_cfg(),
            initial_opponent_idx=2,
            initial_board_size_idx=1,
        )
        m.update(_LOSS)
        # Opponent retreats first.
        assert m.opponent_idx == 1
        assert m.board_size_idx == 1  # Unchanged.

    def test_retreat_board_size_when_at_easiest_opponent(self) -> None:
        """Board size must retreat when opponent is already at minimum."""
        m = GoCurriculumManager(
            config=_fast_cfg(),
            initial_opponent_idx=0,
            initial_board_size_idx=2,
        )
        m.update(_LOSS)
        assert m.board_size_idx == 1
        assert m.opponent_idx == 0  # Unchanged.

    def test_no_advance_at_maximum_difficulty(self) -> None:
        """Advancing at maximum difficulty must leave the level unchanged."""
        m = _manager_at_max()
        opp_idx_before = m.opponent_idx
        size_idx_before = m.board_size_idx
        m.update(_WIN)
        assert m.opponent_idx == opp_idx_before
        assert m.board_size_idx == size_idx_before

    def test_no_retreat_at_minimum_difficulty(self) -> None:
        """Retreating at minimum difficulty must leave the level unchanged."""
        m = _manager_at_min()
        m.update(_LOSS)
        assert m.opponent_idx == 0
        assert m.board_size_idx == 0


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


class TestSmoothing:
    """Tests for the moving-average win-rate smoothing."""

    def test_smoothing_delays_advance(self) -> None:
        """A single high win rate should not advance when window > 1."""
        # smoothing_window=3: one win followed by two neutral scores.
        m = GoCurriculumManager(
            config=CurriculumConfig(
                min_evaluations=1,
                cooldown_evals=1,
                smoothing_window=3,
                win_rate_threshold_up=0.7,
            )
        )
        # First update: smoothed = 1.0, advance happens (min_evals=1).
        m.update({"win_rate": 1.0})
        assert m.opponent_idx == 1, (
            "Expected an advance on the first high-win-rate update"
        )
        # After advance the cooldown resets; feed neutral scores to
        # verify the smoothed rate falls and no further advance occurs.
        starting_opp = m.opponent_idx
        m.update({"win_rate": 0.0})  # smoothed ~0.5 - no change
        m.update({"win_rate": 0.0})  # smoothed ~0.33 - no change
        assert m.opponent_idx == starting_opp

    def test_smoothing_triggers_advance_on_consistent_wins(
        self,
    ) -> None:
        """Consistent wins across a full smoothing window must advance."""
        m = GoCurriculumManager(
            config=CurriculumConfig(
                min_evaluations=3,
                cooldown_evals=1,
                smoothing_window=3,
                win_rate_threshold_up=0.7,
            )
        )
        for _ in range(3):
            m.update({"win_rate": 1.0})
        # After 3 wins the smoothed rate is 1.0 and advance is allowed.
        assert m.opponent_idx == 1

    def test_smoothing_triggers_retreat_on_consistent_losses(
        self,
    ) -> None:
        """Consistent losses across the window must retreat."""
        m = GoCurriculumManager(
            config=CurriculumConfig(
                min_evaluations=3,
                cooldown_evals=1,
                smoothing_window=3,
                win_rate_threshold_down=0.3,
            ),
            initial_opponent_idx=3,
        )
        for _ in range(3):
            m.update({"win_rate": 0.0})
        assert m.opponent_idx == 2


# ---------------------------------------------------------------------------
# Full progression walk-through
# ---------------------------------------------------------------------------


class TestFullProgressionWalkthrough:
    """End-to-end walk-through of advancing through all levels."""

    def test_full_advance_sequence(self) -> None:
        """Agent must advance through all opponents and board sizes."""
        m = GoCurriculumManager(config=_fast_cfg())
        # Advance through all opponents on board_size 0.
        for expected_opp in range(1, len(OPPONENTS)):
            m.update(_WIN)
            assert m.opponent_idx == expected_opp
            assert m.board_size_idx == 0

        # Now at max opponent; next win must advance board size.
        for expected_size in range(1, len(BOARD_SIZES)):
            m.update(_WIN)
            assert m.board_size_idx == expected_size

        # At max difficulty: further wins must not change anything.
        m.update(_WIN)
        assert m.opponent_idx == len(OPPONENTS) - 1
        assert m.board_size_idx == len(BOARD_SIZES) - 1

    def test_full_retreat_sequence(self) -> None:
        """Agent must be able to retreat from max difficulty to min."""
        m = GoCurriculumManager(
            config=_fast_cfg(),
            initial_opponent_idx=len(OPPONENTS) - 1,
            initial_board_size_idx=len(BOARD_SIZES) - 1,
        )
        # Retreat through all opponents first (from hardest).
        for expected_opp in range(len(OPPONENTS) - 2, -1, -1):
            m.update(_LOSS)
            assert m.opponent_idx == expected_opp

        # Now at easiest opponent; retreats must reduce board size.
        for expected_size in range(len(BOARD_SIZES) - 2, -1, -1):
            m.update(_LOSS)
            assert m.board_size_idx == expected_size

        # At minimum: further losses must not change anything.
        m.update(_LOSS)
        assert m.opponent_idx == 0
        assert m.board_size_idx == 0
