"""WebSocket client for communicating with the Bitburner IPvGO engine."""

import asyncio
import json
import logging
from typing import Any

import websockets

logger = logging.getLogger(__name__)


class GoClient:
    """Synchronous WebSocket client wrapping the Bitburner IPvGO engine API.

    All public methods are synchronous and block until the server responds,
    making them safe to call from TorchRL's synchronous ``_reset`` /
    ``_step`` methods.  Each call opens a fresh connection so that the
    client can be used across multiple threads without sharing state.

    Server message contract
    -----------------------
    reset        →
        send: ``{"type": "reset", "opponent": "<opponent>",
                 "board_size": <int>}``
        recv: ``{"success": <bool>}``

        ``<opponent>`` is the bot name (e.g. ``"easy"``) when the game
        will be driven by a built-in IPvGO bot, or ``"no-ai"`` when the
        Python side controls both players (self-play / evaluation).
        ``<board_size>`` is the side length of the square board (e.g. 5,
        7, 9, or 13).

    move         →
        send: ``{"type": "move", "action": <int>}``
        recv: ``{"success": <bool>}``

    observe      →
        send: ``{"type": "observe"}``
        recv: ``{"board": <list[str]>,
                 "current_player": "black"|"white"|"none",
                 "legal_moves": <list[bool]>}``

        ``"current_player"`` is ``"none"`` when the game has ended.

    reward       →
        send: ``{"type": "reward", "player": "black"|"white"}``
        recv: ``{"reward": <int>}``

    move_builtin →
        send: ``{"type": "move_builtin"}``
        recv: ``{"action": <int>}``

        The server instructs the configured built-in bot to play its
        move, advances the game state, and returns the action index the
        bot chose.  No separate ``move`` call is needed afterwards.
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

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send *payload* to the server and return the parsed response.

        Opens a fresh WebSocket connection, sends the JSON-serialised
        payload, reads one response message, and closes the connection.
        Using a new event loop per call keeps the client thread-safe
        when TorchRL creates multiple environment workers.

        Args:
            payload: JSON-serialisable mapping to send to the server.

        Returns:
            Parsed JSON response from the server.
        """
        async def _do() -> dict[str, Any]:
            async with websockets.connect(self.uri) as ws:  # type: ignore[attr-defined]
                logger.debug("send: %s", payload)
                await ws.send(json.dumps(payload))
                raw = await ws.recv()
            response: dict[str, Any] = json.loads(raw)
            logger.debug("recv: %s", response)
            return response

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_do())
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        opponent: str = "no-ai",
        board_size: int = 9,
    ) -> bool:
        """Ask the server to start a new game.

        Args:
            opponent: Name of the opponent the server should use for
                this game.  Pass a built-in bot name (e.g. ``"easy"``,
                ``"medium"``, ``"hard"``) when the game will be driven
                by an IPvGO built-in AI, or ``"no-ai"`` (the default)
                when the Python side controls both players (self-play
                or model-vs-model evaluation).
            board_size: Side length of the square board to request.
                Defaults to ``9``.  Common values are ``5``, ``7``,
                ``9``, and ``13``.

        Returns:
            ``True`` if the server successfully reset the game.

        Raises:
            ValueError: If the server response is missing the
                ``"success"`` key.
        """
        response = self._request(
            {
                "type": "reset",
                "opponent": opponent,
                "board_size": board_size,
            }
        )
        if "success" not in response:
            raise ValueError(
                f"reset: malformed server response, missing 'success' key: "
                f"{response!r}"
            )
        return bool(response["success"])

    def move(self, action: int) -> bool:
        """Send *action* to the server and advance the game state.

        Args:
            action: Integer action index.  Values in ``[0, board_size**2)``
                place a stone; the last index (``board_size**2``) is PASS.

        Returns:
            ``True`` if the server accepted the move.

        Raises:
            ValueError: If the server response is missing the
                ``"success"`` key.
        """
        response = self._request({"type": "move", "action": action})
        if "success" not in response:
            raise ValueError(
                f"move: malformed server response, missing 'success' key: "
                f"{response!r}"
            )
        return bool(response["success"])

    def observe(self) -> dict[str, Any]:
        """Fetch the current game state from the server.

        Returns:
            Server response dict with keys:

            * ``"board"``          - list of board-size strings encoding
              stone positions (``'X'`` black, ``'O'`` white, ``'.'``
              empty).
            * ``"current_player"`` - ``"black"``, ``"white"``, or
              ``"none"`` (game over).
            * ``"legal_moves"``    - flat boolean list of length
              ``board_size * board_size + 1``; the last entry is PASS
              legality.

        Raises:
            ValueError: If any required key is absent from the response.
        """
        response = self._request({"type": "observe"})
        for key in ("board", "current_player", "legal_moves"):
            if key not in response:
                raise ValueError(
                    f"observe: malformed server response, missing '{key}' "
                    f"key: {response!r}"
                )
        return response

    def reward(self, player: str) -> int:
        """Fetch the current reward for *player* from the server.

        Args:
            player: ``"black"`` or ``"white"``.

        Returns:
            Integer reward signal from the server's perspective.

        Raises:
            ValueError: If the server response is missing the
                ``"reward"`` key.
        """
        response = self._request({"type": "reward", "player": player})
        if "reward" not in response:
            raise ValueError(
                f"reward: malformed server response, missing 'reward' key: "
                f"{response!r}"
            )
        return int(response["reward"])

    def move_builtin(self) -> int:
        """Tell the server to have the configured built-in bot play its move.

        The server instructs the built-in bot (set when the game was
        reset) to choose and execute its move, advances the game state,
        and returns the integer action index the bot played.  No
        separate :meth:`move` call is needed afterwards.

        Returns:
            Integer action index the built-in bot played.

        Raises:
            ValueError: If the server response is missing the
                ``"action"`` key.
        """
        response = self._request({"type": "move_builtin"})
        if "action" not in response:
            raise ValueError(
                f"move_builtin: malformed server response, missing 'action' "
                f"key: {response!r}"
            )
        return int(response["action"])
