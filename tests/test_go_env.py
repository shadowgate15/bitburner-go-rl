"""Tests for the TorchRL Go environment and GoServer."""

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import websockets

from src.env.client import GoServer
from src.env.go_env import TorchRLGoEnv, encode_board

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


class TestGoServerLazy:
    """Tests for the lazy GoServer creation inside TorchRLGoEnv."""

    def test_client_not_created_on_init(self) -> None:
        """The server must not be created until first access."""
        env = TorchRLGoEnv(board_size=BOARD_SIZE)
        assert env._client is None

    def test_client_created_on_access(self) -> None:
        """Accessing the client property should create a GoServer."""
        env = TorchRLGoEnv(
            board_size=BOARD_SIZE,
            websocket_uri="ws://unused:9999",
        )
        client = env.client
        assert client is not None
        assert client.host == "unused"
        assert client.port == 9999
        # Second access should return the same object.
        assert env.client is client

    def test_client_prebuilt_is_used_directly(self) -> None:
        """A pre-built GoServer passed at init must be used as-is."""
        from unittest.mock import MagicMock

        mock_server = MagicMock()
        env = TorchRLGoEnv(
            board_size=BOARD_SIZE,
            client=mock_server,
        )
        assert env._client is mock_server
        assert env.client is mock_server


# ---------------------------------------------------------------------------
# GoServer tests
# ---------------------------------------------------------------------------


class TestGoServer:
    """Integration tests for the GoServer WebSocket server."""

    def _find_free_port(self) -> int:
        """Return an OS-assigned free TCP port."""
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def _make_reset_response(self) -> str:
        return json.dumps(
            {
                "board": ["." * BOARD_SIZE] * BOARD_SIZE,
                "current_player": "black",
                "legal_moves": [True] * (BOARD_SIZE * BOARD_SIZE + 1),
            }
        )

    def _make_step_response(self) -> str:
        return json.dumps(
            {
                "board": ["." * BOARD_SIZE] * BOARD_SIZE,
                "current_player": "white",
                "legal_moves": [True] * (BOARD_SIZE * BOARD_SIZE + 1),
                "reward": 0.0,
                "done": False,
            }
        )

    def test_is_not_connected_before_client(self) -> None:
        """Server must report disconnected before any client connects."""
        port = self._find_free_port()
        server = GoServer(host="127.0.0.1", port=port)
        server.start()
        assert server.is_connected is False

    def test_wait_for_client_timeout(self) -> None:
        """wait_for_client must return False when it times out."""
        port = self._find_free_port()
        server = GoServer(host="127.0.0.1", port=port)
        server.start()
        connected = server.wait_for_client(timeout=0.1)
        assert connected is False

    def test_is_connected_after_client_connects(self) -> None:
        """is_connected must be True while a client is connected."""
        port = self._find_free_port()
        server = GoServer(host="127.0.0.1", port=port)
        server.start()

        async def _connect_and_hold() -> None:
            uri = f"ws://127.0.0.1:{port}"
            async with websockets.connect(uri):
                await asyncio.sleep(0.2)

        import threading

        done = threading.Event()

        def _run() -> None:
            asyncio.run(_connect_and_hold())
            done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        connected = server.wait_for_client(timeout=2.0)
        assert connected is True
        assert server.is_connected is True
        done.wait(timeout=2.0)

    def test_reset_and_step_exchange(self) -> None:
        """reset() and step() must successfully exchange messages."""
        port = self._find_free_port()
        server = GoServer(host="127.0.0.1", port=port)
        server.start()

        reset_resp = self._make_reset_response()
        step_resp = self._make_step_response()

        async def _fake_client() -> None:
            uri = f"ws://127.0.0.1:{port}"
            async with websockets.connect(uri) as ws:
                # Respond to reset
                msg = json.loads(await ws.recv())
                assert msg["type"] == "reset"
                await ws.send(reset_resp)
                # Respond to step
                msg = json.loads(await ws.recv())
                assert msg["type"] == "step"
                await ws.send(step_resp)

        import threading

        t = threading.Thread(
            target=lambda: asyncio.run(_fake_client()), daemon=True
        )
        t.start()
        server.wait_for_client(timeout=2.0)

        result = server.reset(opponent="Netburners", board_size=BOARD_SIZE)
        assert "board" in result
        assert "current_player" in result

        step_result = server.step(0)
        assert "reward" in step_result
        assert "done" in step_result

        t.join(timeout=2.0)

    def test_connection_error_when_no_client(self) -> None:
        """reset() must raise ConnectionError if no client is connected."""
        port = self._find_free_port()
        server = GoServer(host="127.0.0.1", port=port)
        server.start()
        with pytest.raises(ConnectionError):
            server.reset()

    def test_is_disconnected_after_client_leaves(self) -> None:
        """is_connected must become False after the client disconnects."""
        port = self._find_free_port()
        server = GoServer(host="127.0.0.1", port=port)
        server.start()

        async def _connect_and_close() -> None:
            uri = f"ws://127.0.0.1:{port}"
            async with websockets.connect(uri):
                pass  # close immediately

        import threading

        t = threading.Thread(
            target=lambda: asyncio.run(_connect_and_close()), daemon=True
        )
        t.start()
        server.wait_for_client(timeout=2.0)
        t.join(timeout=2.0)
        # Give the server handler time to notice the close.
        import time

        time.sleep(0.2)
        assert server.is_connected is False
