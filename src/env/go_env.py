"""TorchRL environment for the board game Go via the Bitburner IPvGO engine."""

from typing import Any

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.data.tensor_specs import Binary, Categorical
from torchrl.envs import EnvBase

from src.env.client import GoClient

# Stone symbols used in the board strings returned by the server.
_BLACK = "X"
_WHITE = "O"


def encode_board(
    board_state: list[str],
    legal_moves: list[bool],
    current_player: str,
    board_size: int,
) -> torch.Tensor:
    """Convert a raw board state into a (4, board_size, board_size) tensor.

    The four channels encode:

    * Channel 0 - black stones (1 where a black stone is present, else 0).
    * Channel 1 - white stones (1 where a white stone is present, else 0).
    * Channel 2 - current-player plane (all 1s if it is black's turn, else 0).
    * Channel 3 - legal-move mask (1 where a move is legal, else 0).
      The last element of *legal_moves* corresponds to the PASS action and is
      placed in channel 3 at position ``(board_size-1, board_size-1)`` when
      not encoded as an extra board cell.  However, since the action space
      includes ``board_size*board_size + 1`` actions (the last being PASS),
      only the first ``board_size*board_size`` elements of *legal_moves* are
      mapped onto the board grid; the PASS legality is not encoded in the
      spatial tensor.

    Args:
        board_state: List of *board_size* strings, each of length *board_size*.
            Characters: ``'X'`` = black stone, ``'O'`` = white stone,
            ``'.'`` = empty, ``'#'`` = dead/void cell.
        legal_moves: Flat boolean list of length
            ``board_size * board_size + 1``.  Each entry at index
            ``row * board_size + col`` indicates whether placing a stone at
            ``(row, col)`` is legal.  The last entry is PASS legality.
        current_player: ``"black"`` or ``"white"``.
        board_size: Side length of the board.

    Returns:
        Float32 tensor of shape ``(4, board_size, board_size)``.
    """
    black = torch.zeros(board_size, board_size, dtype=torch.float32)
    white = torch.zeros(board_size, board_size, dtype=torch.float32)
    legal = torch.zeros(board_size, board_size, dtype=torch.float32)

    for row_idx, row in enumerate(board_state):
        for col_idx, cell in enumerate(row):
            if cell == _BLACK:
                black[row_idx, col_idx] = 1.0
            elif cell == _WHITE:
                white[row_idx, col_idx] = 1.0

    # Map the flat legal-moves list onto the board grid (ignoring PASS slot).
    board_flat = legal_moves[: board_size * board_size]
    for flat_idx, is_legal in enumerate(board_flat):
        row_idx = flat_idx // board_size
        col_idx = flat_idx % board_size
        legal[row_idx, col_idx] = float(is_legal)

    # Current-player plane: all 1s for black, all 0s for white.
    player_plane = torch.ones(board_size, board_size, dtype=torch.float32)
    if current_player != "black":
        player_plane = torch.zeros(board_size, board_size, dtype=torch.float32)

    return torch.stack([black, white, player_plane, legal], dim=0)


def decode_observation(
    obs: torch.Tensor,
) -> tuple[list[str], str, list[bool]]:
    """Reconstruct a server-compatible state dict from an encoded observation.

    This is the inverse of :func:`encode_board`.  It recovers board
    strings, the current player, and the flat legal-move list from a
    4-channel observation tensor.

    Channel decoding:

    * Channel 0 - positions containing ``'X'`` (black stone).
    * Channel 1 - positions containing ``'O'`` (white stone).
    * Channel 2 - if ``obs[2, 0, 0] > 0.5`` the current player is
      ``"black"``; otherwise ``"white"``.
    * Channel 3 - legal board positions; PASS legality is not stored in
      the tensor (see :func:`encode_board`), so it is always appended as
      ``True`` (PASS is always legal in this implementation).

    Limitation:
        The void/dead cell marker ``'#'`` is **not** preserved by
        :func:`encode_board` - those cells appear as empty ``'.'`` after
        decoding.

    Args:
        obs: Float32 tensor of shape ``(4, B, B)`` or ``(1, 4, B, B)``
            as returned by :func:`encode_board` or
            :meth:`~src.env.go_env.TorchRLGoEnv._reset` /
            :meth:`~src.env.go_env.TorchRLGoEnv._step`.

    Returns:
        A 3-tuple ``(board_state, current_player, legal_moves)`` where:

        * ``board_state`` - list of *B* strings each of length *B*,
          using ``'X'``, ``'O'``, and ``'.'``.
        * ``current_player`` - ``"black"`` or ``"white"``.
        * ``legal_moves`` - flat ``bool`` list of length
          ``B * B + 1``; the last element is always ``True`` (PASS).
    """
    if obs.dim() == 4:
        obs = obs[0]  # strip batch dimension

    board_size = obs.shape[-1]

    # Channels 0 and 1: stone positions.
    board: list[str] = []
    for row in range(board_size):
        row_chars: list[str] = []
        for col in range(board_size):
            if obs[0, row, col].item() > 0.5:
                row_chars.append(_BLACK)
            elif obs[1, row, col].item() > 0.5:
                row_chars.append(_WHITE)
            else:
                row_chars.append(".")
        board.append("".join(row_chars))

    # Channel 2: current-player plane.
    current_player = "black" if obs[2, 0, 0].item() > 0.5 else "white"

    # Channel 3: legal-move mask for board positions.
    # PASS (last action) is always legal and is not stored in channel 3.
    legal_moves: list[bool] = [
        obs[3, row, col].item() > 0.5
        for row in range(board_size)
        for col in range(board_size)
    ]
    legal_moves.append(True)  # PASS is always legal

    return board, current_player, legal_moves


class TorchRLGoEnv(EnvBase):
    """TorchRL environment that wraps the Bitburner IPvGO WebSocket engine.

    The environment does **not** implement any Go rules itself.  Every game
    logic decision (legality checking, capture resolution, scoring, …) is
    delegated to the remote Bitburner server through the :class:`GoClient`.

    Observation space
    -----------------
    A float32 tensor of shape ``(4, board_size, board_size)`` with channels:

    * 0 - black stones
    * 1 - white stones
    * 2 - current-player plane
    * 3 - legal-move mask

    Action space
    ------------
    Discrete with ``board_size * board_size + 1`` actions.  The last action
    (index ``board_size * board_size``) represents PASS.

    Args:
        board_size: Side length of the square board (default 9).
        websocket_uri: URI of the Bitburner IPvGO WebSocket server.
    """

    def __init__(
        self,
        board_size: int = 9,
        websocket_uri: str = "ws://localhost:8765",
    ) -> None:
        """Initialise the environment and its specs.

        Args:
            board_size: Side length of the board.
            websocket_uri: WebSocket URI to connect to.
        """
        super().__init__()

        self.board_size = board_size
        self.websocket_uri = websocket_uri

        # Lazy client - created on first use so the environment object can be
        # constructed without a running server (useful for spec inspection).
        self._client: GoClient | None = None

        self._make_specs()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def client(self) -> GoClient:
        """Return the :class:`GoClient`, creating it on first access."""
        if self._client is None:
            self._client = GoClient(self.websocket_uri)
        return self._client

    # ------------------------------------------------------------------
    # Spec construction
    # ------------------------------------------------------------------

    def _make_specs(self) -> None:
        """Build observation, action, reward, and done specs."""
        n_actions = self.board_size * self.board_size + 1
        obs_shape = (4, self.board_size, self.board_size)

        # Observation: 4-channel float board tensor in [0, 1].
        self.observation_spec = Composite(
            observation=Bounded(
                low=0.0,
                high=1.0,
                shape=obs_shape,
                dtype=torch.float32,
            ),
            shape=(),
        )

        # Action: discrete index over board positions + PASS.
        self.action_spec = Categorical(
            n=n_actions, shape=(), dtype=torch.int64
        )

        # Reward: scalar, unbounded.
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)

        # Done flag: scalar boolean.
        self.done_spec = Binary(shape=(1,), dtype=torch.bool)

    # ------------------------------------------------------------------
    # TorchRL interface
    # ------------------------------------------------------------------

    def _reset(
        self,
        tensordict: TensorDict | None = None,
        opponent: str = "no-ai",
    ) -> TensorDict:
        """Reset the environment and return the initial observation.

        Sends a ``reset`` message to the Bitburner server, receives the
        initial board state, and returns a :class:`~tensordict.TensorDict`
        containing the encoded observation and ``done=False``.

        Args:
            tensordict: Unused; present for API compatibility with
                TorchRL's :class:`~torchrl.envs.EnvBase`.
            opponent: Opponent name to pass to the server.  Use a
                built-in bot name (e.g. ``"easy"``, ``"medium"``,
                ``"hard"``) when the server should control the opposing
                player, or ``"no-ai"`` (the default) when the Python
                side controls both players.

        Returns:
            TensorDict with keys ``"observation"`` and ``"done"``.
        """
        response = self.client.reset(opponent, self.board_size)

        board_state: list[str] = response["board"]
        current_player: str = response.get("current_player", "black")
        legal_moves: list[bool] = response["legal_moves"]

        obs = encode_board(
            board_state, legal_moves, current_player, self.board_size
        )

        return TensorDict(
            {
                "observation": obs,
                "done": torch.tensor([False], dtype=torch.bool),
            },
            batch_size=[],
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Advance the environment by one step.

        Extracts the action from *tensordict*, sends it to the Bitburner
        server, and returns a new :class:`~tensordict.TensorDict` containing
        the updated observation, reward, and done flag.

        Args:
            tensordict: Input dict that **must** contain key ``"action"``
                (a scalar integer tensor).

        Returns:
            TensorDict with keys ``"observation"``, ``"reward"``, and
            ``"done"``.
        """
        action: int = int(tensordict["action"].item())

        response = self.client.step(action)

        board_state: list[str] = response["board"]
        reward: float = float(response["reward"])
        done: bool = bool(response["done"])
        current_player: str = response.get("current_player", "black")
        legal_moves: list[bool] = response["legal_moves"]

        obs = encode_board(
            board_state, legal_moves, current_player, self.board_size
        )

        return TensorDict(
            {
                "observation": obs,
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done], dtype=torch.bool),
            },
            batch_size=[],
        )

    def _set_seed(self, seed: int | None) -> None:
        """Set the random seed (no-op - game logic lives on the server).

        Args:
            seed: Ignored.
        """

    # ------------------------------------------------------------------
    # Extra helpers exposed for testing / debugging
    # ------------------------------------------------------------------

    def _encode_step_response(
        self, response: dict[str, Any]
    ) -> TensorDict:
        """Encode a server step response into a TensorDict.

        Converts a raw server response dict into the same
        :class:`~tensordict.TensorDict` format returned by
        :meth:`_step`, but **without** making a WebSocket call.

        This is used by :func:`~src.league.rollout.play_episode` when a
        :class:`~src.league.opponents.BuiltinOpponent` has already
        driven the server to advance game state via
        :meth:`~src.env.client.GoClient.builtin_step`.  In that case
        the caller possesses the server's response dict but must
        **not** send another WebSocket message (the state is already
        advanced).

        Args:
            response: Server response dict as returned by
                :meth:`~src.env.client.GoClient.builtin_step` or
                :meth:`~src.env.client.GoClient.step`.  Must contain
                keys ``"board"``, ``"reward"``, ``"done"``,
                ``"current_player"``, and ``"legal_moves"``.

        Returns:
            TensorDict with keys ``"observation"``, ``"reward"``, and
            ``"done"``, shaped identically to what :meth:`_step` returns.
        """
        board_state: list[str] = response["board"]
        reward: float = float(response["reward"])
        done: bool = bool(response["done"])
        current_player: str = response.get("current_player", "black")
        legal_moves: list[bool] = response["legal_moves"]

        obs = encode_board(
            board_state, legal_moves, current_player, self.board_size
        )

        return TensorDict(
            {
                "observation": obs,
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done], dtype=torch.bool),
            },
            batch_size=[],
        )

    def encode_board(
        self,
        board_state: list[str],
        legal_moves: list[bool],
        current_player: str,
    ) -> torch.Tensor:
        """Convenience wrapper around the module-level :func:`encode_board`.

        Args:
            board_state: Raw board strings from the server.
            legal_moves: Flat boolean legality list.
            current_player: ``"black"`` or ``"white"``.

        Returns:
            Encoded float32 tensor ``(4, board_size, board_size)``.
        """
        return encode_board(
            board_state, legal_moves, current_player, self.board_size
        )

    def get_action_mask(self, legal_moves: list[bool]) -> torch.Tensor:
        """Return a boolean action-mask tensor of shape ``(n_actions,)``.

        Args:
            legal_moves: Flat boolean list of length
                ``board_size * board_size + 1``.

        Returns:
            Boolean tensor of shape ``(board_size * board_size + 1,)``.
        """
        return torch.tensor(legal_moves, dtype=torch.bool)


__all__ = ["TorchRLGoEnv", "decode_observation", "encode_board"]
