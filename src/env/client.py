"""WebSocket client for communicating with the Bitburner IPvGO engine."""

import asyncio
import json
from typing import Any

import websockets


class GoClient:
    """Synchronous WebSocket client wrapping the Bitburner IPvGO engine API.

    All public methods are synchronous and block until the server responds,
    making them safe to call from TorchRL's synchronous ``_reset`` /
    ``_step`` methods.  Each call opens a fresh connection so that the
    client can be used across multiple threads without sharing state.

    Server message contract
    -----------------------
    reset        →
        send: ``{"type": "reset", "opponent": "<opponent>"}``
        recv: ``{"board": <list[str]>, "current_player": "black"|"white",
                 "legal_moves": <list[bool]>}``

        ``<opponent>`` is the bot name (e.g. ``"easy"``) when the game
        will be driven by a built-in IPvGO bot, or ``"no-ai"`` when the
        Python side controls both players (self-play / evaluation).

    step         →
        send: ``{"type": "step", "action": <int>}``
        recv: ``{"board": <list[str]>, "reward": <float>, "done": <bool>,
                 "current_player": "black"|"white",
                 "legal_moves": <list[bool]>}``

    builtin_step →
        send: ``{"type": "builtin_step", "bot": "<bot_name>"}``
        recv: ``{"action": <int>, "board": <list[str]>,
                 "reward": <float>, "done": <bool>,
                 "current_player": "black"|"white",
                 "legal_moves": <list[bool]>}``

        The server instructs the named built-in bot to play its move,
        advances the game state, and returns both the action the bot
        chose and the resulting game state.  No separate ``step`` call
        is needed afterwards.
    """

    def __init__(self, uri: str = "ws://localhost:8765") -> None:
        """Initialise the client.

        Args:
            uri: WebSocket URI of the Bitburner IPvGO server.
        """
        self.uri = uri

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _send_recv(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Open a connection, send *payload*, receive one message, return it.

        Args:
            payload: JSON-serialisable mapping to send to the server.

        Returns:
            Parsed JSON response from the server.
        """
        async with websockets.connect(self.uri) as ws:  # type: ClientConnection
            await ws.send(json.dumps(payload))
            raw = await ws.recv()
        return json.loads(raw)  # type: ignore[return-value]

    def _run(self, coro: Any) -> Any:
        """Execute *coro* in a new event loop and return the result.

        Using a new loop per call keeps the client thread-safe when TorchRL
        creates multiple environment workers.

        Args:
            coro: Awaitable coroutine to execute.

        Returns:
            The return value of *coro*.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, opponent: str = "no-ai") -> dict[str, Any]:
        """Ask the server to start a new game and return the initial state.

        Args:
            opponent: Name of the opponent the server should use for
                this game.  Pass a built-in bot name (e.g. ``"easy"``,
                ``"medium"``, ``"hard"``) when the game will be driven
                by an IPvGO built-in AI, or ``"no-ai"`` (the default)
                when the Python side controls both players (self-play
                or model-vs-model evaluation).

        Returns:
            Server response dict with keys ``board``, ``current_player``,
            and ``legal_moves``.
        """
        return self._run(  # type: ignore[return-value]
            self._send_recv({"type": "reset", "opponent": opponent})
        )

    def step(self, action: int) -> dict[str, Any]:
        """Send *action* to the server and return the updated game state.

        Args:
            action: Integer action index.  Values in ``[0, board_size**2)``
                place a stone; the last index (``board_size**2``) is PASS.

        Returns:
            Server response dict with keys ``board``, ``reward``, ``done``,
            ``current_player``, and ``legal_moves``.
        """
        return self._run(  # type: ignore[return-value]
            self._send_recv({"type": "step", "action": action})
        )

    def builtin_step(self, bot_name: str) -> dict[str, Any]:
        """Tell the server to have a built-in bot play its move.

        Sends a ``builtin_step`` request to the server.  The server
        instructs the named bot to choose and play its move, advances
        the game state, and returns the resulting state together with
        the action index the bot chose.

        Unlike :meth:`step`, **no action is supplied by the caller**:
        move selection is entirely server-driven.  The caller must
        **not** send a subsequent :meth:`step` for the same turn, as
        the game state is already advanced when this method returns.

        Args:
            bot_name: Name of the built-in bot, e.g. ``"easy"``,
                ``"medium"``, or ``"hard"``.

        Returns:
            Server response dict with keys:

            * ``"action"`` - integer action index the bot played.
            * ``"board"`` - updated board strings.
            * ``"reward"`` - float reward from the bot's perspective.
            * ``"done"`` - boolean episode-termination flag.
            * ``"current_player"`` - ``"black"`` or ``"white"``.
            * ``"legal_moves"`` - updated flat boolean legality list.

        Raises:
            NotImplementedError: Always - the server-side API for
                built-in bot steps has not yet been finalised.

        Todo:
            Implement once the WebSocket server exposes the
            ``builtin_step`` message type.  Expected wire format::

                send: {"type": "builtin_step", "bot": "<bot_name>"}
                recv: {
                    "action": <int>,
                    "board": <list[str]>,
                    "reward": <float>,
                    "done": <bool>,
                    "current_player": "black"|"white",
                    "legal_moves": <list[bool]>
                }
        """
        # TODO: implement once WebSocket API details are confirmed.
        raise NotImplementedError(
            f"builtin_step is not yet implemented for bot '{bot_name}'.  "
            f"The WebSocket server API for built-in bot steps needs to be "
            f"defined and deployed."
        )
