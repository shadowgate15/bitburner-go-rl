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
    reset → send: ``{"type": "reset"}``
           recv: ``{"board": <list[str]>, "current_player": "black"|"white",
                    "legal_moves": <list[bool]>}``

    step  → send: ``{"type": "step", "action": <int>}``
           recv: ``{"board": <list[str]>, "reward": <float>, "done": <bool>,
                    "current_player": "black"|"white",
                    "legal_moves": <list[bool]>}``
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

    def reset(self) -> dict[str, Any]:
        """Ask the server to start a new game and return the initial state.

        Returns:
            Server response dict with keys ``board``, ``current_player``,
            and ``legal_moves``.
        """
        return self._run(self._send_recv({"type": "reset"}))  # type: ignore[return-value]

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

    def get_builtin_move(
        self, bot_name: str, state: dict[str, Any]
    ) -> int:
        """Request a move from a built-in IPvGO bot via the server.

        Sends a ``builtin_move`` request to the server, asking the
        named bot to choose an action given the current game *state*.
        The server should respond with a single ``action`` integer.

        Args:
            bot_name: Name of the built-in bot, e.g. ``"easy"``,
                ``"medium"``, or ``"hard"``.
            state: Current game state dict as returned by
                :meth:`reset` or :meth:`step`.  Must contain at
                minimum ``"board"``, ``"current_player"``, and
                ``"legal_moves"``.

        Returns:
            Integer action index chosen by the bot.

        Raises:
            NotImplementedError: Always - the server-side API for
                built-in bot moves has not yet been finalised.

        Todo:
            Implement once the WebSocket server exposes a
            ``builtin_move`` message type.  Expected wire format::

                send: {
                    "type": "builtin_move",
                    "bot": "<bot_name>",
                    "board": <list[str]>,
                    "current_player": "black"|"white",
                    "legal_moves": <list[bool]>
                }
                recv: {"action": <int>}
        """
        # TODO: implement once WebSocket API details are confirmed.
        raise NotImplementedError(
            f"get_builtin_move is not yet implemented for bot "
            f"'{bot_name}'.  The WebSocket server API for built-in "
            f"bot moves needs to be defined and deployed."
        )
