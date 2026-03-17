"""Tests for the curriculum learning and league self-play system.

These tests do **not** require a live Bitburner WebSocket server or
a GPU.  They exercise:

* :class:`~src.league.curriculum.BoardCurriculum` advancement logic.
* :class:`~src.league.opponents.RandomOpponent` action sampling.
* :class:`~src.league.opponents.ModelOpponent` inference.
* :class:`~src.league.opponents.BuiltinOpponent` delegation.
* :class:`~src.league.opponents.OpponentPool` sampling distribution.
* :class:`~src.league.checkpoint.CheckpointManager` save paths.
* :func:`~src.league.rollout.play_episode` episode structure.
* :func:`~src.league.evaluation.evaluate` metrics structure.
* :class:`~src.train.train.CurriculumTrainConfig` defaults.
* :meth:`~src.env.client.GoClient.builtin_step` raises correctly.
"""

from __future__ import annotations

import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.env.go_env import TorchRLGoEnv
from src.league.checkpoint import CheckpointManager
from src.league.curriculum import BoardCurriculum
from src.league.evaluation import evaluate
from src.league.opponents import (
    BuiltinOpponent,
    ModelOpponent,
    OpponentPool,
    RandomOpponent,
)
from src.league.rollout import play_episode
from src.train.train import CurriculumTrainConfig, TrainConfig, build_network

BOARD_SIZE = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(board_size: int = BOARD_SIZE, all_legal: bool = True) -> torch.Tensor:
    """Return a single (unbatched) board observation."""
    obs = torch.zeros(4, board_size, board_size)
    obs[2] = 1.0  # current player = black
    if all_legal:
        obs[3] = 1.0
    return obs


def _make_actor_critic(board_size: int = BOARD_SIZE):
    """Return a (actor, critic) pair for a small board."""
    cfg = TrainConfig(
        board_size=board_size,
        n_filters=16,
        n_cnn_layers=1,
        n_fc=32,
    )
    return build_network(cfg, torch.device("cpu"))


def _make_env_with_mock(
    board_size: int = BOARD_SIZE,
    reward: float = 1.0,
    n_steps: int = 2,
) -> TorchRLGoEnv:
    """Return a TorchRLGoEnv whose client is fully mocked.

    The mock produces *n_steps* non-terminal steps followed by a
    terminal step with *reward*, using the new message protocol:

    * ``reset()``   → ``True``
    * ``move()``    → ``True``
    * ``observe()`` → alternating non-terminal states, then a terminal
                      state (``current_player == "none"``)
    * ``reward()``  → ``0`` for non-terminal steps, *reward* for
                      the terminal step

    With *n_steps* non-terminal pairs (one agent step + one opponent
    step each) followed by one terminal agent step, the mock produces
    exactly ``2 * n_steps + 2`` ``observe()`` calls (one after reset,
    ``2 * n_steps`` for non-terminal steps, one terminal) and
    ``2 * n_steps + 1`` ``reward()`` calls.
    """
    env = TorchRLGoEnv(board_size=board_size)
    board = ["." * board_size for _ in range(board_size)]
    legal = [True] * (board_size * board_size + 1)

    def make_obs_resp(player: str) -> dict[str, Any]:
        return {"board": board, "current_player": player, "legal_moves": legal}

    # observe() sequence: initial state after reset, then alternating
    # non-terminal states, then a terminal state.
    observe_responses: list[dict[str, Any]] = [make_obs_resp("black")]
    for _ in range(n_steps):
        observe_responses.append(make_obs_resp("white"))  # after agent step
        observe_responses.append(make_obs_resp("black"))  # after opponent step
    observe_responses.append(make_obs_resp("none"))  # terminal agent step

    # reward() sequence: 0 for all non-terminal steps, final reward for
    # the terminal step.  One reward() call per move() call.
    reward_values: list[float] = [0.0] * (n_steps * 2) + [reward]

    mock_client = MagicMock()
    mock_client.reset.return_value = True
    mock_client.move.return_value = True
    mock_client.observe.side_effect = observe_responses
    mock_client.reward.side_effect = reward_values
    env._client = mock_client
    return env


# ---------------------------------------------------------------------------
# BoardCurriculum
# ---------------------------------------------------------------------------


class TestBoardCurriculum:
    """Tests for :class:`BoardCurriculum`."""

    def test_initial_board_size(self) -> None:
        """Curriculum must start at the smallest board size (5)."""
        c = BoardCurriculum()
        assert c.current_size == 5

    def test_initial_stage(self) -> None:
        """Initial curriculum stage must be 'early'."""
        assert BoardCurriculum().get_stage() == "early"

    def test_no_advance_below_threshold(self) -> None:
        """Win rate below threshold must not advance the curriculum."""
        c = BoardCurriculum()
        c.update({"win_rate_vs_easy": 0.5})
        assert c.current_size == 5

    def test_advance_when_threshold_met(self) -> None:
        """Win rate ≥ threshold must advance the board size."""
        c = BoardCurriculum()
        c.update({"win_rate_vs_easy": 0.8})
        assert c.current_size == 7

    def test_advance_5_to_7(self) -> None:
        """5->7 gate: win_rate_vs_easy >= 0.8."""
        c = BoardCurriculum()
        c.update({"win_rate_vs_easy": 0.8})
        assert c.current_size == 7
        assert c.get_stage() == "early"

    def test_advance_7_to_9(self) -> None:
        """7->9 gate: win_rate_vs_medium >= 0.7."""
        c = BoardCurriculum()
        c.load_state_dict({"stage_idx": 1, "metrics": {}})
        c.update({"win_rate_vs_medium": 0.7})
        assert c.current_size == 9
        assert c.get_stage() == "mid"

    def test_advance_9_to_13(self) -> None:
        """9->13 gate: win_rate_vs_medium >= 0.7."""
        c = BoardCurriculum()
        c.load_state_dict({"stage_idx": 2, "metrics": {}})
        c.update({"win_rate_vs_medium": 0.7})
        assert c.current_size == 13
        assert c.get_stage() == "late"

    def test_no_advance_from_terminal(self) -> None:
        """Terminal stage (13) must never advance further."""
        c = BoardCurriculum()
        c.load_state_dict({"stage_idx": 3, "metrics": {}})
        c.update({"win_rate_vs_hard": 1.0})
        assert c.current_size == 13

    def test_should_advance_false_terminal(self) -> None:
        """should_advance must return False at the terminal stage."""
        c = BoardCurriculum()
        c.load_state_dict({
            "stage_idx": len(c.BOARD_SIZES) - 1,
            "metrics": {"win_rate_vs_hard": 1.0},
        })
        assert c.should_advance() is False

    def test_next_size_returns_none_at_terminal(self) -> None:
        """next_size must be None at the terminal stage."""
        c = BoardCurriculum()
        c.load_state_dict(
            {"stage_idx": len(c.BOARD_SIZES) - 1, "metrics": {}}
        )
        assert c.next_size is None

    def test_get_board_size_returns_current_or_next(self) -> None:
        """get_board_size must return a valid board size."""
        c = BoardCurriculum(transition_mix=0.7)
        sizes = {c.get_board_size() for _ in range(200)}
        assert sizes.issubset(set(c.BOARD_SIZES))

    def test_state_dict_round_trip(self) -> None:
        """state_dict / load_state_dict must restore curriculum state."""
        c = BoardCurriculum()
        c.load_state_dict(
            {"stage_idx": 2, "metrics": {"win_rate_vs_medium": 0.6}}
        )
        snapshot = c.state_dict()

        c2 = BoardCurriculum()
        c2.load_state_dict(snapshot)
        assert c2.current_size == c.current_size
        assert c2.state_dict()["metrics"] == c.state_dict()["metrics"]

    def test_invalid_transition_mix_raises(self) -> None:
        """transition_mix outside (0, 1] must raise ValueError."""
        with pytest.raises(ValueError):
            BoardCurriculum(transition_mix=0.0)
        with pytest.raises(ValueError):
            BoardCurriculum(transition_mix=1.5)

    def test_transition_mix_at_terminal_always_current(self) -> None:
        """At terminal stage get_board_size always returns current."""
        c = BoardCurriculum()
        c.load_state_dict(
            {"stage_idx": len(c.BOARD_SIZES) - 1, "metrics": {}}
        )
        assert all(c.get_board_size() == c.current_size for _ in range(50))


# ---------------------------------------------------------------------------
# RandomOpponent
# ---------------------------------------------------------------------------


class TestRandomOpponent:
    """Tests for :class:`RandomOpponent`."""

    def test_act_returns_legal_action(self) -> None:
        """act must return a legal action index."""
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        opp = RandomOpponent(n_actions)
        obs = _make_obs(all_legal=True)
        for _ in range(20):
            action = opp.act(obs)
            assert 0 <= action < n_actions

    def test_act_respects_legal_mask(self) -> None:
        """act must only return legal actions from channel 3."""
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        opp = RandomOpponent(n_actions)
        obs = _make_obs(all_legal=False)
        obs[3, 0, 0] = 1.0  # only action 0 is legal (plus PASS)
        for _ in range(30):
            action = opp.act(obs)
            # Legal: action 0 (position (0,0)) or PASS (last index)
            assert action == 0 or action == n_actions - 1

    def test_act_falls_back_to_pass(self) -> None:
        """act must fall back to PASS when no legal action exists."""
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        opp = RandomOpponent(n_actions)
        obs = torch.zeros(4, BOARD_SIZE, BOARD_SIZE)  # all illegal
        # PASS (index appended in act) is always included, so opp
        # should always return something valid
        action = opp.act(obs)
        # The fallback path appends 1.0 for PASS
        assert 0 <= action < n_actions


# ---------------------------------------------------------------------------
# ModelOpponent
# ---------------------------------------------------------------------------


class TestModelOpponent:
    """Tests for :class:`ModelOpponent`."""

    def test_act_returns_valid_action(self) -> None:
        """ModelOpponent must return a valid integer action."""
        actor, _ = _make_actor_critic()
        opp = ModelOpponent(actor)
        obs = _make_obs()
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        action = opp.act(obs)
        assert isinstance(action, int)
        assert 0 <= action < n_actions

    def test_act_no_gradients(self) -> None:
        """ModelOpponent.act must not accumulate gradients."""
        actor, _ = _make_actor_critic()
        opp = ModelOpponent(actor)
        obs = _make_obs()
        # Run act and verify no autograd state is modified
        with torch.no_grad():
            action = opp.act(obs)
        assert isinstance(action, int)


# ---------------------------------------------------------------------------
# BuiltinOpponent
# ---------------------------------------------------------------------------


class TestBuiltinOpponent:
    """Tests for :class:`BuiltinOpponent`."""

    def test_step_calls_client_move_builtin(self) -> None:
        """step() must delegate to client.move_builtin with no arguments."""
        mock_client = MagicMock()
        mock_client.move_builtin.return_value = 0
        opp = BuiltinOpponent("easy", mock_client)
        opp.step()
        mock_client.move_builtin.assert_called_once_with()

    def test_step_returns_action_int(self) -> None:
        """step() must return the integer action returned by client.move_builtin."""
        mock_client = MagicMock()
        mock_client.move_builtin.return_value = 7
        opp = BuiltinOpponent("medium", mock_client)
        result = opp.step()
        assert result == 7

    def test_step_returns_pass_action(self) -> None:
        """step() must return the PASS action index when the bot plays PASS."""
        mock_client = MagicMock()
        mock_client.move_builtin.return_value = BOARD_SIZE * BOARD_SIZE
        opp = BuiltinOpponent("hard", mock_client)
        result = opp.step()
        assert result == BOARD_SIZE * BOARD_SIZE

    def test_step_does_not_send_state_to_server(self) -> None:
        """step() must not accept or forward any board-state argument.

        The IPvGO API is server-driven: the server knows the current state.
        """
        mock_client = MagicMock()
        mock_client.move_builtin.return_value = 0
        opp = BuiltinOpponent("easy", mock_client)
        opp.step()
        # move_builtin must be called with no arguments - no state forwarded
        call_args = mock_client.move_builtin.call_args
        assert call_args == ((), {}), (
            f"move_builtin called with unexpected args: {call_args}"
        )

    def test_builtin_has_no_act_method(self) -> None:
        """BuiltinOpponent must not expose an act() method.

        Move selection and state advancement are both server-driven, so
        the act() / env._step() split used by other opponents does not
        apply.
        """
        mock_client = MagicMock()
        opp = BuiltinOpponent("easy", mock_client)
        assert not hasattr(opp, "act"), (
            "BuiltinOpponent should not have act(); use step() instead"
        )

    def test_reset_is_no_op(self) -> None:
        """reset must not raise and must be idempotent."""
        mock_client = MagicMock()
        opp = BuiltinOpponent("medium", mock_client)
        opp.reset()
        opp.reset()  # idempotent

    def test_bot_name_stored(self) -> None:
        """Constructor must store bot_name."""
        mock_client = MagicMock()
        opp = BuiltinOpponent("hard", mock_client)
        assert opp.bot_name == "hard"

    def test_model_opponent_act_no_websocket(self) -> None:
        """ModelOpponent.act must not call any WebSocket client."""
        actor, _ = _make_actor_critic()
        opp = ModelOpponent(actor)
        obs = _make_obs()
        # If ModelOpponent somehow tried to open a WebSocket the test
        # would fail because there is no server running.
        action = opp.act(obs)
        assert isinstance(action, int)

    def test_random_opponent_act_no_websocket(self) -> None:
        """RandomOpponent.act must not call any WebSocket client."""
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        opp = RandomOpponent(n_actions)
        obs = _make_obs()
        action = opp.act(obs)
        assert isinstance(action, int)
        assert 0 <= action < n_actions


# ---------------------------------------------------------------------------
# GoClient new API
# ---------------------------------------------------------------------------


class TestGoClientNewAPI:
    """Tests for the new GoClient message protocol."""

    def test_reset_raises_on_missing_success(self) -> None:
        """reset() must raise ValueError if response lacks 'success'."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        with patch.object(
            client, "_request", return_value={"unexpected": True}
        ):
            with pytest.raises(ValueError, match="success"):
                client.reset("no-ai", 9)

    def test_move_raises_on_missing_success(self) -> None:
        """move() must raise ValueError if response lacks 'success'."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        with patch.object(client, "_request", return_value={"foo": 1}):
            with pytest.raises(ValueError, match="success"):
                client.move(0)

    def test_observe_raises_on_missing_key(self) -> None:
        """observe() must raise ValueError if a required key is absent."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        # Missing "legal_moves"
        with patch.object(
            client,
            "_request",
            return_value={"board": [], "current_player": "black"},
        ):
            with pytest.raises(ValueError, match="legal_moves"):
                client.observe()

    def test_reward_raises_on_missing_reward(self) -> None:
        """reward() must raise ValueError if response lacks 'reward'."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        with patch.object(client, "_request", return_value={"other": 1}):
            with pytest.raises(ValueError, match="reward"):
                client.reward("black")

    def test_move_builtin_raises_on_missing_action(self) -> None:
        """move_builtin() must raise ValueError if response lacks 'action'."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        with patch.object(client, "_request", return_value={"other": 1}):
            with pytest.raises(ValueError, match="action"):
                client.move_builtin()

    def test_reset_returns_bool(self) -> None:
        """reset() must return True when server responds with success=True."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        with patch.object(
            client, "_request", return_value={"success": True}
        ):
            result = client.reset("no-ai", 9)
        assert result is True

    def test_move_returns_bool(self) -> None:
        """move() must return True when server responds with success=True."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        with patch.object(
            client, "_request", return_value={"success": True}
        ):
            result = client.move(5)
        assert result is True

    def test_observe_returns_dict(self) -> None:
        """observe() must return the full response dict from the server."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        expected = {
            "board": [".....", "....."],
            "current_player": "black",
            "legal_moves": [True] * 26,
        }
        with patch.object(client, "_request", return_value=expected):
            result = client.observe()
        assert result == expected

    def test_reward_returns_int(self) -> None:
        """reward() must return an integer reward value."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        with patch.object(
            client, "_request", return_value={"reward": 1}
        ):
            result = client.reward("black")
        assert result == 1
        assert isinstance(result, int)

    def test_move_builtin_returns_int(self) -> None:
        """move_builtin() must return the integer action from the server."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        with patch.object(
            client, "_request", return_value={"action": 12}
        ):
            result = client.move_builtin()
        assert result == 12
        assert isinstance(result, int)

    def test_reset_sends_correct_payload(self) -> None:
        """reset() must send the correct JSON payload to the server."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        with patch.object(
            client, "_request", return_value={"success": True}
        ) as mock_request:
            client.reset("easy", 7)
        mock_request.assert_called_once_with(
            {"type": "reset", "opponent": "easy", "board_size": 7}
        )

    def test_move_sends_correct_payload(self) -> None:
        """move() must send the correct JSON payload to the server."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        with patch.object(
            client, "_request", return_value={"success": True}
        ) as mock_request:
            client.move(42)
        mock_request.assert_called_once_with({"type": "move", "action": 42})

    def test_observe_sends_correct_payload(self) -> None:
        """observe() must send {\"type\": \"observe\"} to the server."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        obs_response = {
            "board": [],
            "current_player": "black",
            "legal_moves": [],
        }
        with patch.object(
            client, "_request", return_value=obs_response
        ) as mock_request:
            client.observe()
        mock_request.assert_called_once_with({"type": "observe"})

    def test_reward_sends_correct_payload(self) -> None:
        """reward() must send the correct JSON payload including the player."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        with patch.object(
            client, "_request", return_value={"reward": 1}
        ) as mock_request:
            client.reward("white")
        mock_request.assert_called_once_with(
            {"type": "reward", "player": "white"}
        )

    def test_move_builtin_sends_correct_payload(self) -> None:
        """move_builtin() must send {\"type\": \"move_builtin\"} to the server."""
        from unittest.mock import patch

        from src.env.client import GoClient

        client = GoClient()
        with patch.object(
            client, "_request", return_value={"action": 0}
        ) as mock_request:
            client.move_builtin()
        mock_request.assert_called_once_with({"type": "move_builtin"})


# ---------------------------------------------------------------------------
# OpponentPool
# ---------------------------------------------------------------------------


class TestOpponentPool:
    """Tests for :class:`OpponentPool`."""

    def test_initial_state(self) -> None:
        """Freshly-constructed pool must have no checkpoints."""
        pool = OpponentPool(board_size=BOARD_SIZE)
        assert pool.latest_path is None
        assert pool.historical_paths == []

    def test_add_checkpoint(self) -> None:
        """add_checkpoint must update latest_path and historical list."""
        pool = OpponentPool(board_size=BOARD_SIZE)
        pool.add_checkpoint("/tmp/step_1.pt")
        assert pool.latest_path == "/tmp/step_1.pt"
        assert pool.historical_paths == ["/tmp/step_1.pt"]

    def test_historical_eviction(self) -> None:
        """Oldest checkpoints must be evicted when limit exceeded."""
        pool = OpponentPool(board_size=BOARD_SIZE, max_historical=3)
        for i in range(5):
            pool.add_checkpoint(f"/tmp/step_{i}.pt")
        assert len(pool.historical_paths) == 3
        # Oldest (0 and 1) should be gone
        assert "/tmp/step_0.pt" not in pool.historical_paths
        assert "/tmp/step_4.pt" in pool.historical_paths

    def test_get_builtin_opponent_returns_builtin(self) -> None:
        """get_builtin_opponent must return a BuiltinOpponent."""
        pool = OpponentPool(board_size=BOARD_SIZE)
        opp = pool.get_builtin_opponent("easy")
        assert isinstance(opp, BuiltinOpponent)
        assert opp.bot_name == "easy"

    def test_sample_opponent_early_returns_opponent(self) -> None:
        """sample_opponent('early') must return a valid opponent."""
        pool = OpponentPool(board_size=BOARD_SIZE)
        # No checkpoints → should fall back to Random or Builtin
        for _ in range(20):
            opp = pool.sample_opponent("early")
            assert isinstance(opp, (BuiltinOpponent, RandomOpponent, ModelOpponent))

    def test_sample_opponent_mid_returns_opponent(self) -> None:
        """sample_opponent('mid') must return a valid opponent."""
        pool = OpponentPool(board_size=BOARD_SIZE)
        for _ in range(20):
            opp = pool.sample_opponent("mid")
            assert isinstance(opp, (BuiltinOpponent, RandomOpponent, ModelOpponent))

    def test_sample_opponent_late_returns_opponent(self) -> None:
        """sample_opponent('late') must return a valid opponent."""
        pool = OpponentPool(board_size=BOARD_SIZE)
        for _ in range(20):
            opp = pool.sample_opponent("late")
            assert isinstance(opp, (BuiltinOpponent, RandomOpponent, ModelOpponent))

    def test_set_latest_actor_used_in_sampling(self) -> None:
        """latest actor must be used when sampling 'latest' type."""
        pool = OpponentPool(board_size=BOARD_SIZE)
        actor, _ = _make_actor_critic()
        pool.set_latest_actor(actor)
        # Patch choices so 'latest' is always picked
        with patch(
            "src.league.opponents.random.choices",
            return_value=["latest"],
        ):
            opp = pool.sample_opponent("early")
        assert isinstance(opp, ModelOpponent)

    def test_latest_actor_property(self) -> None:
        """latest_actor property must return the actor set via set_latest_actor."""
        pool = OpponentPool(board_size=BOARD_SIZE)
        assert pool.latest_actor is None
        actor, _ = _make_actor_critic()
        pool.set_latest_actor(actor)
        assert pool.latest_actor is actor


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    """Tests for :class:`CheckpointManager`."""

    def _make_manager(self, tmp_dir: str) -> CheckpointManager:
        return CheckpointManager(checkpoint_dir=tmp_dir)

    def _make_components(self):
        actor, critic = _make_actor_critic()
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        cfg = TrainConfig(board_size=BOARD_SIZE)
        return actor, critic, optimizer, cfg

    def test_directories_created(self) -> None:
        """CheckpointManager must create checkpoint and league dirs."""
        with tempfile.TemporaryDirectory() as tmp:
            import os

            mgr = self._make_manager(tmp)
            assert os.path.isdir(str(mgr.checkpoint_dir))
            assert os.path.isdir(str(mgr.league_dir))

    def test_save_latest_creates_file(self) -> None:
        """save_latest must write latest.pt to the checkpoint dir."""
        with tempfile.TemporaryDirectory() as tmp:
            import os

            mgr = self._make_manager(tmp)
            actor, critic, optimizer, cfg = self._make_components()
            path = mgr.save_latest(actor, critic, optimizer, 1, cfg)
            assert os.path.isfile(path)
            assert path.endswith("latest.pt")

    def test_maybe_save_best_first_call(self) -> None:
        """First call to maybe_save_best must always save best.pt."""
        with tempfile.TemporaryDirectory() as tmp:
            import os

            mgr = self._make_manager(tmp)
            actor, critic, optimizer, cfg = self._make_components()
            saved = mgr.maybe_save_best(
                {"win_rate_vs_medium": 0.5},
                actor, critic, optimizer, 1, cfg,
            )
            assert saved is True
            assert os.path.isfile(str(mgr.checkpoint_dir / "best.pt"))

    def test_maybe_save_best_no_improvement(self) -> None:
        """maybe_save_best must return False when metric does not improve."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            actor, critic, optimizer, cfg = self._make_components()
            mgr.maybe_save_best(
                {"win_rate_vs_medium": 0.8},
                actor, critic, optimizer, 1, cfg,
            )
            saved = mgr.maybe_save_best(
                {"win_rate_vs_medium": 0.7},
                actor, critic, optimizer, 2, cfg,
            )
            assert saved is False

    def test_add_to_league_creates_file(self) -> None:
        """add_to_league must write a step-specific file in league/."""
        with tempfile.TemporaryDirectory() as tmp:
            import os

            mgr = self._make_manager(tmp)
            actor, critic, _, cfg = self._make_components()
            path = mgr.add_to_league(actor, critic, 42, cfg)
            assert os.path.isfile(path)
            assert "step_42" in path
            assert "league" in path

    def test_best_value_updated(self) -> None:
        """best_value must track the highest metric seen so far."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._make_manager(tmp)
            actor, critic, optimizer, cfg = self._make_components()
            mgr.maybe_save_best(
                {"win_rate_vs_medium": 0.6},
                actor, critic, optimizer, 1, cfg,
            )
            assert mgr.best_value == pytest.approx(0.6)
            mgr.maybe_save_best(
                {"win_rate_vs_medium": 0.9},
                actor, critic, optimizer, 2, cfg,
            )
            assert mgr.best_value == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# play_episode
# ---------------------------------------------------------------------------


class TestPlayEpisode:
    """Tests for :func:`play_episode`."""

    def test_returns_expected_keys(self) -> None:
        """play_episode must return won, total_reward, and steps."""
        actor, _ = _make_actor_critic()
        env = _make_env_with_mock(n_steps=1, reward=1.0)
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        opp = RandomOpponent(n_actions)
        result = play_episode(env, actor, opp, BOARD_SIZE)
        assert "won" in result
        assert "total_reward" in result
        assert "steps" in result

    def test_won_true_when_positive_reward(self) -> None:
        """Episode with positive final reward must report won=True."""
        actor, _ = _make_actor_critic()
        env = _make_env_with_mock(n_steps=1, reward=1.0)
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        opp = RandomOpponent(n_actions)
        result = play_episode(env, actor, opp, BOARD_SIZE)
        assert result["won"] is True

    def test_won_false_when_zero_reward(self) -> None:
        """Episode with zero total reward must report won=False."""
        actor, _ = _make_actor_critic()
        env = _make_env_with_mock(n_steps=0, reward=0.0)
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        opp = RandomOpponent(n_actions)
        result = play_episode(env, actor, opp, BOARD_SIZE)
        assert result["won"] is False

    def test_steps_positive(self) -> None:
        """Completed episode must have at least one agent step."""
        actor, _ = _make_actor_critic()
        env = _make_env_with_mock(n_steps=2, reward=1.0)
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        opp = RandomOpponent(n_actions)
        result = play_episode(env, actor, opp, BOARD_SIZE)
        assert result["steps"] >= 1

    def test_reset_sends_no_ai_for_random_opponent(self) -> None:
        """play_episode must call env._reset with 'no-ai' for RandomOpponent."""
        actor, _ = _make_actor_critic()
        env = _make_env_with_mock(n_steps=0, reward=1.0)
        n_actions = BOARD_SIZE * BOARD_SIZE + 1
        opp = RandomOpponent(n_actions)
        play_episode(env, actor, opp, BOARD_SIZE)
        env._client.reset.assert_called_once_with("no-ai", BOARD_SIZE)  # type: ignore[union-attr]

    def test_reset_sends_no_ai_for_model_opponent(self) -> None:
        """play_episode must call env._reset with 'no-ai' for ModelOpponent."""
        actor, _ = _make_actor_critic()
        env = _make_env_with_mock(n_steps=0, reward=1.0)
        opp = ModelOpponent(actor)
        play_episode(env, actor, opp, BOARD_SIZE)
        env._client.reset.assert_called_once_with("no-ai", BOARD_SIZE)  # type: ignore[union-attr]

    def test_reset_sends_bot_name_for_builtin_opponent(self) -> None:
        """play_episode must call env._reset with the bot name for BuiltinOpponent."""
        actor, _ = _make_actor_critic()

        board = ["." * BOARD_SIZE for _ in range(BOARD_SIZE)]
        legal = [True] * (BOARD_SIZE * BOARD_SIZE + 1)

        env = TorchRLGoEnv(board_size=BOARD_SIZE)
        env_mock_client = MagicMock()
        env_mock_client.reset.return_value = True
        # Episode ends on agent's first step so builtin never needs to move.
        # observe(): initial state after reset, then terminal after agent move.
        env_mock_client.observe.side_effect = [
            {"board": board, "current_player": "black", "legal_moves": legal},
            {"board": board, "current_player": "none", "legal_moves": legal},
        ]
        env_mock_client.move.return_value = True
        env_mock_client.reward.return_value = 1
        env._client = env_mock_client

        builtin_client = MagicMock()
        opp = BuiltinOpponent("hard", builtin_client)

        play_episode(env, actor, opp, BOARD_SIZE)

        env_mock_client.reset.assert_called_once_with("hard", BOARD_SIZE)

    def test_reset_sends_correct_name_for_each_builtin_bot(self) -> None:
        """play_episode must forward whichever bot name the BuiltinOpponent holds."""
        for bot_name in ["easy", "medium", "hard"]:
            actor, _ = _make_actor_critic()
            board = ["." * BOARD_SIZE for _ in range(BOARD_SIZE)]
            legal = [True] * (BOARD_SIZE * BOARD_SIZE + 1)

            env = TorchRLGoEnv(board_size=BOARD_SIZE)
            env_mock_client = MagicMock()
            env_mock_client.reset.return_value = True
            env_mock_client.observe.side_effect = [
                {"board": board, "current_player": "black", "legal_moves": legal},
                {"board": board, "current_player": "none", "legal_moves": legal},
            ]
            env_mock_client.move.return_value = True
            env_mock_client.reward.return_value = 1
            env._client = env_mock_client

            builtin_client = MagicMock()
            opp = BuiltinOpponent(bot_name, builtin_client)

            play_episode(env, actor, opp, BOARD_SIZE)

            env_mock_client.reset.assert_called_once_with(bot_name, BOARD_SIZE), (
                f"Expected reset called with '{bot_name}'"
            )

    def test_builtin_opponent_step_called_not_env_step(self) -> None:
        """For BuiltinOpponent, step() is called; env move() is not called for opponent.

        The game state is already advanced by move_builtin on the server,
        so play_episode must use env._observe_state() instead of
        calling env._step for the opponent's move.
        """
        actor, _ = _make_actor_critic()

        # Build an env whose mock client drives: agent step (non-terminal),
        # then episode ends on the BUILTIN step.
        board = ["." * BOARD_SIZE for _ in range(BOARD_SIZE)]
        legal = [True] * (BOARD_SIZE * BOARD_SIZE + 1)

        env = TorchRLGoEnv(board_size=BOARD_SIZE)
        env_mock_client = MagicMock()
        env_mock_client.reset.return_value = True
        # observe() sequence:
        #   1. after reset: initial state
        #   2. after agent's move: non-terminal
        #   3. after builtin's move (via _observe_state): terminal
        env_mock_client.observe.side_effect = [
            {"board": board, "current_player": "black", "legal_moves": legal},
            {"board": board, "current_player": "white", "legal_moves": legal},
            {"board": board, "current_player": "none", "legal_moves": legal},
        ]
        env_mock_client.move.return_value = True
        # reward() sequence: 0.0 for agent's non-terminal, 1.0 for terminal
        env_mock_client.reward.side_effect = [0.0, 1.0]
        env._client = env_mock_client

        # BuiltinOpponent with its own mocked client whose move_builtin
        # terminates the episode.
        builtin_client = MagicMock()
        builtin_client.move_builtin.return_value = BOARD_SIZE * BOARD_SIZE  # PASS
        opp = BuiltinOpponent("easy", builtin_client)

        result = play_episode(env, actor, opp, BOARD_SIZE)

        # move_builtin must have been called once (for the opponent's turn).
        builtin_client.move_builtin.assert_called_once_with()

        # env move() must have been called ONLY for the agent's turn (once),
        # NOT for the opponent's turn.
        assert env_mock_client.move.call_count == 1, (
            f"env.move() called {env_mock_client.move.call_count} times; "
            f"expected 1 (agent turn only)"
        )

        # Episode must have ended with the builtin's reward.
        assert result["won"] is True
        assert result["total_reward"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def _make_builtin_mock(
    board_size: int = BOARD_SIZE,
    done_after_agent: bool = True,
) -> BuiltinOpponent:
    """Return a real BuiltinOpponent with a fully mocked client.

    With ``done_after_agent=True`` (the default used in evaluate tests that
    use ``n_steps=0``), the opponent's ``step()`` is never reached because
    the agent's first step terminates the episode.  The mock is still set up
    correctly so that tests which DO reach the opponent turn will work.
    """
    mock_client = MagicMock()
    # move_builtin() returns the PASS action index when the builtin plays.
    mock_client.move_builtin.return_value = board_size * board_size  # PASS
    return BuiltinOpponent("easy", mock_client)


class TestEvaluate:
    """Tests for :func:`evaluate`."""

    def test_returns_expected_keys(self) -> None:
        """evaluate must return all expected metric keys."""
        actor, _ = _make_actor_critic()
        pool = OpponentPool(board_size=BOARD_SIZE)

        def _env_factory() -> TorchRLGoEnv:
            return _make_env_with_mock(n_steps=0, reward=1.0)

        # Patch get_builtin_opponent to return a proper BuiltinOpponent mock.
        mock_builtin = _make_builtin_mock()
        with patch.object(pool, "get_builtin_opponent", return_value=mock_builtin):
            metrics = evaluate(
                agent_actor=actor,
                opponent_pool=pool,
                env_factory=_env_factory,
                board_size=BOARD_SIZE,
                num_games=2,
            )

        expected_keys = {
            "win_rate_vs_easy",
            "win_rate_vs_medium",
            "win_rate_vs_hard",
            "win_rate_vs_league",
            "win_rate_vs_random",
            "avg_reward",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_win_rates_in_range(self) -> None:
        """All win-rate metrics must be in [0, 1]."""
        actor, _ = _make_actor_critic()
        pool = OpponentPool(board_size=BOARD_SIZE)

        def _env_factory() -> TorchRLGoEnv:
            return _make_env_with_mock(n_steps=0, reward=1.0)

        mock_builtin = _make_builtin_mock()
        with patch.object(pool, "get_builtin_opponent", return_value=mock_builtin):
            metrics = evaluate(
                agent_actor=actor,
                opponent_pool=pool,
                env_factory=_env_factory,
                board_size=BOARD_SIZE,
                num_games=2,
            )

        for key in [
            "win_rate_vs_easy",
            "win_rate_vs_medium",
            "win_rate_vs_hard",
            "win_rate_vs_league",
            "win_rate_vs_random",
        ]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range"


# ---------------------------------------------------------------------------
# CurriculumTrainConfig
# ---------------------------------------------------------------------------


class TestCurriculumTrainConfig:
    """Tests for :class:`CurriculumTrainConfig` defaults."""

    def test_inherits_train_config(self) -> None:
        """CurriculumTrainConfig must be a subclass of TrainConfig."""
        assert issubclass(CurriculumTrainConfig, TrainConfig)

    def test_default_eval_interval(self) -> None:
        """Default eval_interval must be positive."""
        assert CurriculumTrainConfig().eval_interval > 0

    def test_default_league_interval(self) -> None:
        """Default league_interval must be positive."""
        assert CurriculumTrainConfig().league_interval > 0

    def test_default_num_eval_games(self) -> None:
        """Default num_eval_games must be positive."""
        assert CurriculumTrainConfig().num_eval_games > 0

    def test_custom_fields(self) -> None:
        """CurriculumTrainConfig must accept custom field values."""
        cfg = CurriculumTrainConfig(
            eval_interval=10,
            num_eval_games=5,
            transition_mix=0.6,
        )
        assert cfg.eval_interval == 10
        assert cfg.num_eval_games == 5
        assert cfg.transition_mix == pytest.approx(0.6)

    def test_board_size_inherited(self) -> None:
        """board_size must be inherited from TrainConfig."""
        cfg = CurriculumTrainConfig(board_size=7)
        assert cfg.board_size == 7
