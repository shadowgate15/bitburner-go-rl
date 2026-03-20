"""TorchRL environment for the board game Go via the Bitburner IPvGO engine."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.data.tensor_specs import Binary, Categorical
from torchrl.envs import EnvBase

from src.env.client import GoServer

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


class TorchRLGoEnv(EnvBase):
    """TorchRL environment that wraps the Bitburner IPvGO WebSocket engine.

    The environment does **not** implement any Go rules itself.  Every game
    logic decision (legality checking, capture resolution, scoring, …) is
    delegated to the remote Bitburner client through the :class:`GoServer`.

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
        websocket_uri: URI used to derive the host and port when no
            *client* is supplied.  Format: ``ws://host:port``.
        opponent: Built-in BitBurner opponent name (default
            ``"Netburners"``).
        client: Optional pre-built :class:`GoServer` instance to use
            instead of creating one lazily from *websocket_uri*.  Pass
            a shared server when multiple environments must communicate
            over the same connection (e.g. the training and evaluation
            environments in :func:`~src.train.train.train_with_curriculum`).
    """

    def __init__(
        self,
        board_size: int = 9,
        websocket_uri: str = "ws://localhost:8765",
        opponent: str = "Netburners",
        client: GoServer | None = None,
    ) -> None:
        """Initialise the environment and its specs.

        Args:
            board_size: Side length of the board.
            websocket_uri: WebSocket URI used to infer host/port when
                *client* is not provided.
            opponent: Name of the built-in BitBurner opponent to play
                against.  Can be updated at any time by assigning
                ``env.opponent = "..."``; the new value is used on the
                next episode reset.
            client: Pre-built :class:`GoServer` to use.  When ``None``
                a server is created lazily from *websocket_uri* on first
                access.
        """
        super().__init__()

        self.board_size = board_size
        self.websocket_uri = websocket_uri
        #: Current opponent - mutable so the curriculum can update it
        #: between episodes without rebuilding the environment.
        self.opponent: str = opponent

        # Use the provided server directly, or create one lazily on first
        # access via the `client` property.
        self._client: GoServer | None = client

        self._make_specs()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def client(self) -> GoServer:
        """Return the :class:`GoServer`, creating it lazily on first access.

        When no *client* was provided at construction time a new
        :class:`GoServer` is created from :attr:`websocket_uri`.  The
        server is **not** started automatically; the caller (typically the
        training loop) is responsible for calling :meth:`GoServer.start`
        and :meth:`GoServer.wait_for_client` before issuing any game
        commands.
        """
        if self._client is None:
            self._client = GoServer.from_uri(self.websocket_uri)
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

    def rebuild_specs(self) -> None:
        """Rebuild observation, action, reward, and done specs.

        Call this after changing :attr:`board_size` so that the specs
        reflect the new board dimensions.  The curriculum training loop
        calls this automatically when it detects a board-size change.
        """
        self._make_specs()

    # ------------------------------------------------------------------
    # TorchRL interface
    # ------------------------------------------------------------------

    def _reset(
        self,
        tensordict: TensorDict | None = None,
        opponent: str | None = None,
        board_size: int | None = None,
    ) -> TensorDict:
        """Reset the environment and return the initial observation.

        Sends a ``reset`` message to the Bitburner server, receives the
        initial board state, and returns a :class:`~tensordict.TensorDict`
        containing the encoded observation and ``done=False``.

        When *opponent* or *board_size* are provided they override the
        environment's stored values (``self.opponent`` /
        ``self.board_size``).  If *board_size* changes, the observation
        and action specs are rebuilt automatically.

        Args:
            tensordict: Unused; present for API compatibility.
            opponent: Name of the built-in opponent to use for this
                episode.  When ``None``, ``self.opponent`` is used.
            board_size: Side length of the board for this episode.
                When ``None``, ``self.board_size`` is used.

        Returns:
            TensorDict with keys ``"observation"`` and ``"done"``.
        """
        # Apply overrides and persist them on the instance so that
        # subsequent automatic resets (triggered by done=True inside
        # the collector) use the same curriculum settings.
        if opponent is not None:
            self.opponent = opponent
        if board_size is not None and board_size != self.board_size:
            self.board_size = board_size
            self.rebuild_specs()

        response = self.client.reset(
            opponent=self.opponent,
            board_size=self.board_size,
        )

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


__all__ = ["TorchRLGoEnv", "encode_board"]
