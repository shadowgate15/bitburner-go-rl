"""Tests for the TorchRL Go environment and GoClient."""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from src.env.go_env import TorchRLGoEnv, decode_observation, encode_board

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOARD_SIZE = 5  # Use a small board so tests are fast


def _make_board(size: int = BOARD_SIZE) -> list[str]:
    """Return an empty board of *size* x *size* as a list of strings."""
    return ["." * size for _ in range(size)]


def _make_legal_moves(
    size: int = BOARD_SIZE,
    all_legal: bool = True,
) -> list[bool]:
    """Return a flat legal-moves list for a board of *size*."""
    return [all_legal] * (size * size + 1)


def _make_reset_response(size: int = BOARD_SIZE) -> dict[str, Any]:
    """Return a fake server reset response."""
    return {
        "board": _make_board(size),
        "current_player": "black",
        "legal_moves": _make_legal_moves(size),
    }


def _make_step_response(
    size: int = BOARD_SIZE,
    reward: float = 0.0,
    done: bool = False,
) -> dict[str, Any]:
    """Return a fake server step response."""
    return {
        "board": _make_board(size),
        "current_player": "white",
        "legal_moves": _make_legal_moves(size),
        "reward": reward,
        "done": done,
    }


def _make_action_td(action: int = 0) -> Any:
    """Return a TensorDict containing the given action."""
    from tensordict import TensorDict

    return TensorDict(
        {"action": torch.tensor(action, dtype=torch.int64)},
        batch_size=[],
    )


# ---------------------------------------------------------------------------
# encode_board tests
# ---------------------------------------------------------------------------


class TestEncodeBoard:
    """Tests for the stand-alone encode_board helper."""

    def test_output_shape(self) -> None:
        """Encoded tensor must have shape (4, board_size, board_size)."""
        board = _make_board()
        legal = _make_legal_moves()
        obs = encode_board(board, legal, "black", BOARD_SIZE)
        assert obs.shape == (4, BOARD_SIZE, BOARD_SIZE)

    def test_dtype_is_float32(self) -> None:
        """Encoded tensor dtype must be float32."""
        obs = encode_board(
            _make_board(), _make_legal_moves(), "black", BOARD_SIZE
        )
        assert obs.dtype == torch.float32

    def test_empty_board_channels_zero(self) -> None:
        """Channels 0 and 1 must be all zeros on an empty board."""
        obs = encode_board(
            _make_board(), _make_legal_moves(), "black", BOARD_SIZE
        )
        assert obs[0].sum().item() == 0.0  # no black stones
        assert obs[1].sum().item() == 0.0  # no white stones

    def test_black_stone_channel(self) -> None:
        """A black stone at (0,0) should set channel 0 position (0,0) to 1."""
        top_row = "X" + "." * (BOARD_SIZE - 1)
        board = [top_row] + ["." * BOARD_SIZE] * (BOARD_SIZE - 1)
        obs = encode_board(board, _make_legal_moves(), "black", BOARD_SIZE)
        assert obs[0, 0, 0].item() == 1.0
        assert obs[1, 0, 0].item() == 0.0  # no white stone there

    def test_white_stone_channel(self) -> None:
        """A white stone at (1, 2) should set channel 1 at (1, 2) to 1."""
        board = ["." * BOARD_SIZE] * BOARD_SIZE
        row = list(board[1])
        row[2] = "O"
        board[1] = "".join(row)
        obs = encode_board(board, _make_legal_moves(), "white", BOARD_SIZE)
        assert obs[1, 1, 2].item() == 1.0
        assert obs[0, 1, 2].item() == 0.0

    def test_current_player_black(self) -> None:
        """Channel 2 must be all ones when it is black's turn."""
        obs = encode_board(
            _make_board(), _make_legal_moves(), "black", BOARD_SIZE
        )
        assert obs[2].all()

    def test_current_player_white(self) -> None:
        """Channel 2 must be all zeros when it is white's turn."""
        obs = encode_board(
            _make_board(), _make_legal_moves(), "white", BOARD_SIZE
        )
        assert (obs[2] == 0).all()

    def test_legal_move_mask(self) -> None:
        """Channel 3 must reflect the legal-moves list."""
        legal: list[bool] = [False] * (BOARD_SIZE * BOARD_SIZE + 1)
        # Mark only action 0 (row=0, col=0) as legal.
        legal[0] = True
        obs = encode_board(_make_board(), legal, "black", BOARD_SIZE)
        assert obs[3, 0, 0].item() == 1.0
        # All other positions should be 0.
        obs[3, 0, 0] = 0.0
        assert obs[3].sum().item() == 0.0

    def test_all_legal_moves(self) -> None:
        """Channel 3 must be all ones when every move is legal."""
        obs = encode_board(
            _make_board(),
            _make_legal_moves(all_legal=True),
            "black",
            BOARD_SIZE,
        )
        assert obs[3].sum().item() == float(BOARD_SIZE * BOARD_SIZE)


# ---------------------------------------------------------------------------
# TorchRLGoEnv tests
# ---------------------------------------------------------------------------


class TestTorchRLGoEnvSpecs:
    """Tests for TorchRLGoEnv spec shapes."""

    def _make_env(self, size: int = BOARD_SIZE) -> TorchRLGoEnv:
        """Create a TorchRLGoEnv with the given board size."""
        return TorchRLGoEnv(board_size=size)

    def test_observation_spec_shape(self) -> None:
        """Observation spec shape must be (4, board_size, board_size)."""
        env = self._make_env()
        obs_shape = env.observation_spec["observation"].shape
        assert tuple(obs_shape) == (4, BOARD_SIZE, BOARD_SIZE)

    def test_action_spec_n(self) -> None:
        """Action spec must have board_size**2 + 1 discrete actions."""
        env = self._make_env()
        assert env.action_spec.n == BOARD_SIZE * BOARD_SIZE + 1

    def test_reward_spec_shape(self) -> None:
        """Reward spec shape must be (1,)."""
        env = self._make_env()
        assert tuple(env.reward_spec.shape) == (1,)

    def test_done_spec_shape(self) -> None:
        """Inner done Binary spec shape must be (1,)."""
        # TorchRL wraps done_spec in a Composite;
        # the inner Binary has shape (1,).
        env = self._make_env()
        assert tuple(env.done_spec["done"].shape) == (1,)

    def test_custom_board_size(self) -> None:
        """Specs must scale with custom board size."""
        env = TorchRLGoEnv(board_size=13)
        assert env.observation_spec["observation"].shape == (4, 13, 13)
        assert env.action_spec.n == 13 * 13 + 1


class TestTorchRLGoEnvReset:
    """Tests for TorchRLGoEnv._reset."""

    def _make_env_with_mock_client(
        self, size: int = BOARD_SIZE
    ) -> TorchRLGoEnv:
        """Return an env whose client is replaced by a MagicMock."""
        env = TorchRLGoEnv(board_size=size)
        mock_client = MagicMock()
        mock_client.reset.return_value = _make_reset_response(size)
        env._client = mock_client
        return env

    def test_reset_returns_tensordict(self) -> None:
        """_reset must return a TensorDict with observation and done keys."""
        env = self._make_env_with_mock_client()
        td = env._reset()
        assert "observation" in td
        assert "done" in td

    def test_reset_observation_shape(self) -> None:
        """Observation in reset output must have correct spatial shape."""
        env = self._make_env_with_mock_client()
        td = env._reset()
        assert tuple(td["observation"].shape) == (4, BOARD_SIZE, BOARD_SIZE)

    def test_reset_done_is_false(self) -> None:
        """The done flag after reset must be False."""
        env = self._make_env_with_mock_client()
        td = env._reset()
        assert td["done"].item() is False

    def test_reset_calls_client_reset(self) -> None:
        """_reset must call client.reset exactly once."""
        env = self._make_env_with_mock_client()
        env._reset()
        env._client.reset.assert_called_once()  # type: ignore[union-attr]

    def test_reset_default_opponent_is_no_ai(self) -> None:
        """_reset with no explicit opponent must send 'no-ai' to the client."""
        env = self._make_env_with_mock_client()
        env._reset()
        env._client.reset.assert_called_once_with("no-ai", BOARD_SIZE)  # type: ignore[union-attr]

    def test_reset_builtin_opponent_forwarded(self) -> None:
        """_reset must forward the given opponent name to client.reset."""
        env = self._make_env_with_mock_client()
        env._reset(opponent="easy")
        env._client.reset.assert_called_once_with("easy", BOARD_SIZE)  # type: ignore[union-attr]

    def test_reset_no_ai_forwarded_explicitly(self) -> None:
        """_reset must forward 'no-ai' to client.reset when given explicitly."""
        env = self._make_env_with_mock_client()
        env._reset(opponent="no-ai")
        env._client.reset.assert_called_once_with("no-ai", BOARD_SIZE)  # type: ignore[union-attr]

    def test_reset_board_size_forwarded(self) -> None:
        """_reset must forward self.board_size as the board_size arg."""
        env = self._make_env_with_mock_client(size=7)
        env._reset()
        env._client.reset.assert_called_once_with("no-ai", 7)  # type: ignore[union-attr]

    def test_reset_board_size_matches_env(self) -> None:
        """board_size sent to client must equal env.board_size."""
        for size in (5, 7, 9, 13):
            env = TorchRLGoEnv(board_size=size)
            mock_client = MagicMock()
            mock_client.reset.return_value = _make_reset_response(size)
            env._client = mock_client
            env._reset()
            _, called_size = mock_client.reset.call_args.args
            assert called_size == size, f"Expected board_size={size}, got {called_size}"


class TestTorchRLGoEnvStep:
    """Tests for TorchRLGoEnv._step."""

    def _make_env_with_mock_client(
        self,
        size: int = BOARD_SIZE,
        reward: float = 0.0,
        done: bool = False,
    ) -> TorchRLGoEnv:
        """Return an env whose client is replaced by a MagicMock."""
        env = TorchRLGoEnv(board_size=size)
        mock_client = MagicMock()
        mock_client.step.return_value = _make_step_response(
            size, reward, done
        )
        env._client = mock_client
        return env

    def test_step_returns_tensordict(self) -> None:
        """_step must return a TensorDict with obs, reward, done keys."""
        env = self._make_env_with_mock_client()
        td = env._step(_make_action_td(0))
        assert "observation" in td
        assert "reward" in td
        assert "done" in td

    def test_step_observation_shape(self) -> None:
        """Observation in step output must have correct spatial shape."""
        env = self._make_env_with_mock_client()
        td = env._step(_make_action_td(0))
        assert tuple(td["observation"].shape) == (4, BOARD_SIZE, BOARD_SIZE)

    def test_step_reward_value(self) -> None:
        """The reward tensor must reflect the server response value."""
        env = self._make_env_with_mock_client(reward=5.0)
        td = env._step(_make_action_td(0))
        assert td["reward"].item() == pytest.approx(5.0)

    def test_step_done_true(self) -> None:
        """The done flag must be True when the server signals game over."""
        env = self._make_env_with_mock_client(done=True)
        td = env._step(_make_action_td(0))
        assert td["done"].item() is True

    def test_step_done_false(self) -> None:
        """The done flag must be False when the game is still in progress."""
        env = self._make_env_with_mock_client(done=False)
        td = env._step(_make_action_td(0))
        assert td["done"].item() is False

    def test_step_passes_action_to_client(self) -> None:
        """_step must forward the action integer to client.step."""
        env = self._make_env_with_mock_client()
        env._step(_make_action_td(action=7))
        env._client.step.assert_called_once_with(7)  # type: ignore[union-attr]

    def test_pass_action_index(self) -> None:
        """The PASS action (last index) should be forwarded to the client."""
        env = self._make_env_with_mock_client()
        pass_action = BOARD_SIZE * BOARD_SIZE  # last valid action
        env._step(_make_action_td(action=pass_action))
        env._client.step.assert_called_once_with(  # type: ignore[union-attr]
            pass_action
        )


class TestTorchRLGoEnvHelpers:
    """Tests for helper methods on TorchRLGoEnv."""

    def test_get_action_mask_shape(self) -> None:
        """get_action_mask must return a tensor of shape (n_actions,)."""
        env = TorchRLGoEnv(board_size=BOARD_SIZE)
        legal = _make_legal_moves()
        mask = env.get_action_mask(legal)
        assert mask.shape == (BOARD_SIZE * BOARD_SIZE + 1,)
        assert mask.dtype == torch.bool

    def test_get_action_mask_values(self) -> None:
        """get_action_mask values must match the input boolean list."""
        env = TorchRLGoEnv(board_size=BOARD_SIZE)
        legal = [False] * (BOARD_SIZE * BOARD_SIZE + 1)
        legal[3] = True
        mask = env.get_action_mask(legal)
        assert mask[3].item() is True
        assert mask[0].item() is False

    def test_encode_board_method_delegates(self) -> None:
        """The instance method must produce identical results to function."""
        env = TorchRLGoEnv(board_size=BOARD_SIZE)
        board = _make_board()
        legal = _make_legal_moves()
        result_method = env.encode_board(board, legal, "black")
        result_func = encode_board(board, legal, "black", BOARD_SIZE)
        assert torch.equal(result_method, result_func)


class TestGoClientLazy:
    """Tests for the lazy GoClient creation inside TorchRLGoEnv."""

    def test_client_not_created_on_init(self) -> None:
        """The client must not be created until first access."""
        env = TorchRLGoEnv(board_size=BOARD_SIZE)
        assert env._client is None

    def test_client_created_on_access(self) -> None:
        """Accessing the client property should create it."""
        env = TorchRLGoEnv(
            board_size=BOARD_SIZE,
            websocket_uri="ws://unused:9999",
        )
        client = env.client
        assert client is not None
        assert client.uri == "ws://unused:9999"
        # Second access should return the same object.
        assert env.client is client


# ---------------------------------------------------------------------------
# decode_observation tests
# ---------------------------------------------------------------------------


class TestDecodeObservation:
    """Tests for the decode_observation helper (inverse of encode_board)."""

    def _roundtrip(
        self,
        board: list[str],
        legal: list[bool],
        player: str,
    ) -> tuple[list[str], str, list[bool]]:
        """Encode then decode and return the decoded values."""
        obs = encode_board(board, legal, player, BOARD_SIZE)
        return decode_observation(obs)

    def test_empty_board_decoded(self) -> None:
        """Empty board must decode to all '.' characters."""
        board = _make_board()
        legal = _make_legal_moves()
        dec_board, dec_player, dec_legal = self._roundtrip(
            board, legal, "black"
        )
        for row in dec_board:
            assert set(row) == {"."}, f"Unexpected char in row: {row!r}"

    def test_black_stone_roundtrip(self) -> None:
        """A black stone encoded at (0,0) must decode back to 'X' at (0,0)."""
        top_row = "X" + "." * (BOARD_SIZE - 1)
        board = [top_row] + ["." * BOARD_SIZE] * (BOARD_SIZE - 1)
        dec_board, _, _ = self._roundtrip(board, _make_legal_moves(), "black")
        assert dec_board[0][0] == "X"
        # All other cells should be empty.
        assert dec_board[0][1:] == "." * (BOARD_SIZE - 1)

    def test_white_stone_roundtrip(self) -> None:
        """A white stone encoded at (1,2) must decode back to 'O' at (1,2)."""
        board = ["." * BOARD_SIZE] * BOARD_SIZE
        row = list(board[1])
        row[2] = "O"
        board[1] = "".join(row)
        dec_board, _, _ = self._roundtrip(board, _make_legal_moves(), "white")
        assert dec_board[1][2] == "O"

    def test_current_player_black_roundtrip(self) -> None:
        """Current player 'black' must round-trip correctly."""
        _, dec_player, _ = self._roundtrip(
            _make_board(), _make_legal_moves(), "black"
        )
        assert dec_player == "black"

    def test_current_player_white_roundtrip(self) -> None:
        """Current player 'white' must round-trip correctly."""
        _, dec_player, _ = self._roundtrip(
            _make_board(), _make_legal_moves(), "white"
        )
        assert dec_player == "white"

    def test_legal_moves_roundtrip(self) -> None:
        """Legal-move mask must round-trip for board positions."""
        legal = [False] * (BOARD_SIZE * BOARD_SIZE + 1)
        legal[0] = True  # only position (0,0) is legal
        _, _, dec_legal = self._roundtrip(_make_board(), legal, "black")
        # Board positions: only index 0 is True.
        assert dec_legal[0] is True
        assert all(v is False for v in dec_legal[1:BOARD_SIZE * BOARD_SIZE])
        # PASS is always appended as True.
        assert dec_legal[BOARD_SIZE * BOARD_SIZE] is True

    def test_pass_always_legal_in_decoded(self) -> None:
        """PASS (last index) must always be True regardless of input."""
        legal = [False] * (BOARD_SIZE * BOARD_SIZE + 1)
        _, _, dec_legal = self._roundtrip(_make_board(), legal, "black")
        assert dec_legal[-1] is True

    def test_decoded_legal_list_length(self) -> None:
        """Decoded legal_moves must have length board_size**2 + 1."""
        _, _, dec_legal = self._roundtrip(
            _make_board(), _make_legal_moves(), "black"
        )
        assert len(dec_legal) == BOARD_SIZE * BOARD_SIZE + 1

    def test_decoded_board_shape(self) -> None:
        """Decoded board must be a list of board_size strings."""
        dec_board, _, _ = self._roundtrip(
            _make_board(), _make_legal_moves(), "black"
        )
        assert len(dec_board) == BOARD_SIZE
        assert all(len(row) == BOARD_SIZE for row in dec_board)

    def test_batched_input_stripped(self) -> None:
        """decode_observation must handle a (1, 4, B, B) batched tensor."""
        obs = encode_board(
            _make_board(), _make_legal_moves(), "black", BOARD_SIZE
        )
        obs_batched = obs.unsqueeze(0)  # (1, 4, B, B)
        dec_board, dec_player, dec_legal = decode_observation(obs_batched)
        assert len(dec_board) == BOARD_SIZE
        assert dec_player == "black"
        assert len(dec_legal) == BOARD_SIZE * BOARD_SIZE + 1
