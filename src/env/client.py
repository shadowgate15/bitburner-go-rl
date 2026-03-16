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
