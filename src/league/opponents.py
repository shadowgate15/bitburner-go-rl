"""Opponent abstractions for the league self-play system.

This module defines four opponent types used during training and
evaluation:

* :class:`BuiltinOpponent`  - delegates to an IPvGO server-side bot;
  move selection **and** game-state advancement happen server-side.
* :class:`RandomOpponent`   - samples uniformly from legal actions.
* :class:`ModelOpponent`    - wraps a frozen TorchRL actor module.
* :class:`OpponentPool`     - league manager that owns all opponent
  variants and samples them according to the curriculum stage.

:class:`RandomOpponent` and :class:`ModelOpponent` expose an
``act(state)`` method that accepts the current board-observation tensor
and returns an integer action index; game-state advancement is then
handled by the caller via
:meth:`~src.env.go_env.TorchRLGoEnv._step`.

:class:`BuiltinOpponent` exposes a ``step()`` method instead: a single
call both selects the bot's move **and** advances the server-side game
state, so the caller must not call
:meth:`~src.env.go_env.TorchRLGoEnv._step` for that turn.
:func:`~src.league.rollout.play_episode` handles this distinction
automatically.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:  # pragma: no cover
    from src.env.client import GoClient
    from src.train.train import TrainConfig


# ---------------------------------------------------------------------------
# Opponent protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OpponentProtocol(Protocol):
    """Structural interface that every opponent must satisfy."""

    def act(self, state: torch.Tensor) -> int:
        """Choose an action given the current board observation.

        Args:
            state: Float32 tensor of shape
                ``(4, board_size, board_size)``.

        Returns:
            Integer action index.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Opponent implementations
# ---------------------------------------------------------------------------


class BuiltinOpponent:
    """Interface to a named server-side IPvGO bot.

    Delegates move selection **and game-state advancement** to the
    Bitburner IPvGO server by calling
    :meth:`~src.env.client.GoClient.builtin_step`.

    Unlike :class:`RandomOpponent` and :class:`ModelOpponent`, which
    only *choose* an action and leave state advancement to the
    environment's :meth:`~src.env.go_env.TorchRLGoEnv._step` method,
    :class:`BuiltinOpponent` makes a single WebSocket call that both
    selects and plays the bot's move server-side.  The game state is
    therefore **already advanced** when :meth:`step` returns, so the
    caller must **not** call :meth:`~src.env.go_env.TorchRLGoEnv._step`
    for the same turn.

    The opponent is stateless between episodes; call :meth:`reset` at
    the start of each game if any per-episode server state needs to be
    cleared.

    Args:
        bot_name: Server-side bot identifier, e.g. ``"easy"``,
            ``"medium"``, or ``"hard"``.
        client: Initialised :class:`~src.env.client.GoClient` used to
            communicate with the Bitburner server.
    """

    def __init__(self, bot_name: str, client: GoClient) -> None:
        """Initialise the built-in opponent.

        Args:
            bot_name: Name of the server-side bot.
            client: WebSocket client connected to the Bitburner server.
        """
        self.bot_name = bot_name
        self._client = client

    def step(self) -> dict[str, Any]:
        """Tell the server to have the bot play its move.

        Calls :meth:`~src.env.client.GoClient.builtin_step`, which
        instructs the named bot to choose and execute its move on the
        server.  The game state is advanced **server-side** before this
        method returns.

        The caller (:func:`~src.league.rollout.play_episode`) must use
        the returned response to update its local state observation and
        must **not** send a subsequent
        :meth:`~src.env.go_env.TorchRLGoEnv._step` for the same turn.

        Returns:
            Server response dict with keys:

            * ``"action"``         - int action index the bot played.
            * ``"board"``          - updated board strings.
            * ``"reward"``         - float reward signal.
            * ``"done"``           - bool episode-termination flag.
            * ``"current_player"`` - ``"black"`` or ``"white"``.
            * ``"legal_moves"``    - updated flat boolean legality list.

        Note:
            This method calls
            :meth:`~src.env.client.GoClient.builtin_step` which
            raises :exc:`NotImplementedError` until the WebSocket API
            for built-in bot steps is implemented on the server side.
        """
        return self._client.builtin_step(self.bot_name)

    def reset(self) -> None:
        """No-op reset hook.

        :class:`BuiltinOpponent` is stateless - the server manages all
        game state, so nothing needs to be cleared between episodes.
        """


class RandomOpponent:
    """Opponent that samples uniformly at random from legal actions.

    Legal board positions are read from channel 3 of the observation
    tensor (the legal-move mask encoded by
    :func:`~src.env.go_env.encode_board`).  The PASS action (last
    index) is always included in the legal-action set.

    Args:
        n_actions: Total number of actions, i.e.
            ``board_size * board_size + 1``.
    """

    def __init__(self, n_actions: int) -> None:
        """Initialise the random opponent.

        Args:
            n_actions: Total action count including PASS.
        """
        self.n_actions = n_actions

    def act(self, state: torch.Tensor) -> int:
        """Sample a legal action uniformly at random.

        Args:
            state: Float32 board tensor of shape
                ``(4, board_size, board_size)``.

        Returns:
            Integer action index.
        """
        legal_mask: list[float]
        if state.dim() >= 3:
            # Channel 3 holds the spatial legal-move map.
            legal_spatial = state[3] if state.dim() == 3 else state[0, 3]
            legal_mask = legal_spatial.flatten().tolist()
        else:
            legal_mask = [1.0] * (self.n_actions - 1)
        # PASS is always legal.
        legal_mask.append(1.0)
        legal_actions = [
            i for i, v in enumerate(legal_mask) if float(v) > 0.5
        ]
        return random.choice(legal_actions) if legal_actions else (
            self.n_actions - 1
        )


class ModelOpponent:
    """Opponent backed by a frozen TorchRL ``ProbabilisticActor``.

    Wraps a TorchRL actor module so that it can be used wherever an
    :class:`OpponentProtocol` is expected.  All forward passes run
    without gradients (the model is treated as frozen).

    Args:
        actor: Trained TorchRL ``ProbabilisticActor`` (or any callable
            that maps ``{"observation": tensor}`` → ``{"action": tensor}``).
        device: Torch device to run inference on.
    """

    def __init__(
        self,
        actor: Any,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialise the model opponent.

        Args:
            actor: Trained TorchRL actor module.
            device: Device for inference.
        """
        self._actor = actor
        self._device = torch.device(device)

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> int:
        """Select an action using the frozen actor network.

        Args:
            state: Float32 board tensor of shape
                ``(4, board_size, board_size)``.

        Returns:
            Integer action index sampled from the actor's policy.
        """
        from tensordict import TensorDict  # local import to avoid cycle

        obs = state.to(self._device)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)  # (1, C, B, B)
        td = TensorDict({"observation": obs}, batch_size=[obs.shape[0]])
        out = self._actor(td)
        return int(out["action"][0].item())


# ---------------------------------------------------------------------------
# Opponent pool (league manager)
# ---------------------------------------------------------------------------


class OpponentPool:
    """League manager that owns and samples all opponent variants.

    Maintains three opponent categories:

    * **Built-in**: Named IPvGO server bots (easy / medium / hard).
    * **Random**: A uniform-random baseline.
    * **Model**: Torch-based opponents (latest checkpoint + historical).

    The sampling distribution is controlled by the curriculum *stage*:

    * ``"early"``  - 40 % built-in, 30 % random, 20 % latest,
      10 % historical.
    * ``"mid"``    - 25 % built-in, 25 % latest, 25 % historical,
      25 % random.
    * ``"late"``   - 12.5 % built-in, 0 % random, 43.75 % latest,
      43.75 % historical.

    All returned opponents implement :class:`OpponentProtocol`.

    Args:
        websocket_uri: URI used to create
            :class:`~src.env.client.GoClient` instances for built-in
            opponents.
        max_historical: Maximum number of historical checkpoints to
            retain (oldest are evicted when the limit is exceeded).
        board_size: Default board size used by
            :class:`RandomOpponent`.
    """

    BUILTIN_BOT_NAMES: ClassVar[list[str]] = ["easy", "medium", "hard"]

    # Sampling weights per stage: [builtin, random, latest, historical]
    _STAGE_WEIGHTS: ClassVar[dict[str, list[float]]] = {
        "early": [0.40, 0.30, 0.20, 0.10],
        "mid": [0.25, 0.25, 0.25, 0.25],
        "late": [0.125, 0.00, 0.4375, 0.4375],
    }
    _STAGE_KEYS: ClassVar[list[str]] = [
        "builtin", "random", "latest", "historical"
    ]

    def __init__(
        self,
        websocket_uri: str = "ws://localhost:8765",
        max_historical: int = 20,
        board_size: int = 9,
    ) -> None:
        """Initialise the opponent pool.

        Args:
            websocket_uri: WebSocket URI for built-in bot requests.
            max_historical: Maximum historical checkpoints to keep.
            board_size: Board size for the random-opponent fallback.
        """
        self._websocket_uri = websocket_uri
        self._max_historical = max_historical
        self._board_size = board_size

        self._latest_path: str | None = None
        self._historical_paths: list[str] = []
        self._latest_actor: Any = None

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def add_checkpoint(self, path: str) -> None:
        """Register a new checkpoint as the latest model snapshot.

        The checkpoint is also prepended to the historical list.  If
        the list exceeds *max_historical* the oldest entry is removed.

        Args:
            path: File path to the saved checkpoint (``*.pt``).
        """
        self._latest_path = path
        self._historical_paths.append(path)
        if len(self._historical_paths) > self._max_historical:
            self._historical_paths.pop(0)

    def set_latest_actor(self, actor: Any) -> None:
        """Update the in-memory reference to the current policy.

        Args:
            actor: Trained TorchRL actor module (the *live* policy,
                not a checkpoint).
        """
        self._latest_actor = actor

    # ------------------------------------------------------------------
    # Opponent construction
    # ------------------------------------------------------------------

    def get_builtin_opponent(self, name: str) -> BuiltinOpponent:
        """Return a :class:`BuiltinOpponent` for the named server bot.

        Args:
            name: Bot name, e.g. ``"easy"``, ``"medium"``, ``"hard"``.

        Returns:
            A freshly-constructed :class:`BuiltinOpponent`.
        """
        from src.env.client import GoClient  # local import to avoid cycle

        client = GoClient(self._websocket_uri)
        return BuiltinOpponent(name, client)

    def load_model_opponent(
        self, path: str, cfg: TrainConfig | None = None
    ) -> ModelOpponent:
        """Load a checkpoint and wrap it as a :class:`ModelOpponent`.

        The checkpoint must have been saved by
        :class:`~src.league.checkpoint.CheckpointManager` (or the
        training loop in :mod:`src.train.train`) and must contain the
        keys ``"actor_state_dict"`` and ``"cfg"``.

        Args:
            path: File path to the ``.pt`` checkpoint.
            cfg: Optional :class:`~src.train.train.TrainConfig` to
                use when reconstructing the network.  If *None*, the
                config stored inside the checkpoint is used.

        Returns:
            A :class:`ModelOpponent` backed by the loaded actor.
        """
        from src.train.train import TrainConfig, build_network  # local

        data = torch.load(path, map_location="cpu", weights_only=False)
        ckpt_cfg: TrainConfig = (
            data.get("cfg", TrainConfig()) if cfg is None else cfg
        )
        actor, _ = build_network(ckpt_cfg, torch.device("cpu"))
        actor.load_state_dict(data["actor_state_dict"])
        actor.eval()
        return ModelOpponent(actor, device="cpu")

    def _random_opponent(self) -> RandomOpponent:
        """Return a :class:`RandomOpponent` for the current board size.

        Returns:
            A :class:`RandomOpponent` with the pool's board size.
        """
        n_actions = self._board_size * self._board_size + 1
        return RandomOpponent(n_actions)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_opponent(
        self, stage: str
    ) -> BuiltinOpponent | RandomOpponent | ModelOpponent:
        """Sample an opponent appropriate for *stage*.

        Args:
            stage: Curriculum stage label - one of ``"early"``,
                ``"mid"``, or ``"late"``.

        Returns:
            An opponent object implementing :class:`OpponentProtocol`.
            Falls back to :class:`RandomOpponent` when the chosen
            category has no suitable checkpoint.
        """
        weights = self._STAGE_WEIGHTS.get(
            stage, self._STAGE_WEIGHTS["early"]
        )
        chosen = random.choices(self._STAGE_KEYS, weights=weights, k=1)[0]

        if chosen == "builtin":
            return self.get_builtin_opponent(
                random.choice(self.BUILTIN_BOT_NAMES)
            )
        if chosen == "random":
            return self._random_opponent()
        if chosen == "latest":
            if self._latest_actor is not None:
                return ModelOpponent(self._latest_actor, device="cpu")
            if self._latest_path is not None:
                return self.load_model_opponent(self._latest_path)
            return self._random_opponent()
        # "historical"
        if self._historical_paths:
            path = random.choice(self._historical_paths)
            return self.load_model_opponent(path)
        return self._random_opponent()

    # ------------------------------------------------------------------
    # Properties (read-only views)
    # ------------------------------------------------------------------

    @property
    def historical_paths(self) -> list[str]:
        """Ordered list of historical checkpoint paths (oldest first)."""
        return list(self._historical_paths)

    @property
    def latest_path(self) -> str | None:
        """Path to the most-recently added checkpoint, or ``None``."""
        return self._latest_path

    @property
    def latest_actor(self) -> Any:
        """In-memory reference to the current live policy, or ``None``."""
        return self._latest_actor


__all__ = [
    "BuiltinOpponent",
    "ModelOpponent",
    "OpponentPool",
    "OpponentProtocol",
    "RandomOpponent",
]
