"""WebSocket server for communicating with the Bitburner IPvGO engine.

The Bitburner game acts as the WebSocket *client*; this Python server
listens for incoming connections and exchanges game-state messages over
the persistent connection.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import websockets
import websockets.exceptions
from websockets.asyncio.server import ServerConnection


class GoServer:
    """Synchronous wrapper around a WebSocket server for the IPvGO engine.

    The Bitburner game connects *to* this server.  All public methods are
    synchronous and block until the connected client responds, making them
    safe to call from TorchRL's synchronous ``_reset`` / ``_step`` methods.

    The server runs an asyncio event loop in a background daemon thread so
    that the caller never needs to manage the loop directly.

    Message contract (identical to the previous client contract, with
    roles reversed — now Python is the server)
    ------------------------------------------------------------------
    reset → send: ``{"type": "reset", "opponent": …, "board_size": …}``
           recv: ``{"board": …, "current_player": "black"|"white",
                    "legal_moves": …}``

    step  → send: ``{"type": "step", "action": <int>}``
           recv: ``{"board": …, "reward": <float>, "done": <bool>,
                    "current_player": "black"|"white",
                    "legal_moves": …}``
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        """Initialise the server (does not start listening yet).

        Call :meth:`start` to begin accepting connections, then
        :meth:`wait_for_client` to block until the Bitburner game
        connects.

        Args:
            host: Network interface to bind to.  Defaults to
                ``"0.0.0.0"`` (all interfaces).
            port: TCP port to listen on.
        """
        self.host = host
        self.port = port

        # Currently connected client WebSocket (set by handler coroutine).
        self._ws: ServerConnection | None = None

        # Event loop is created inside the background thread (see
        # _run_loop) so it belongs exclusively to that thread.  It is
        # set to None until start() is called.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        # Events for lifecycle synchronisation across threads.
        self._server_ready = threading.Event()
        self._client_connected = threading.Event()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_uri(cls, uri: str) -> GoServer:
        """Create a :class:`GoServer` from a WebSocket URI string.

        Parses ``ws://host:port`` and returns a server that listens on
        the extracted host and port.

        Args:
            uri: WebSocket URI in ``ws://host:port`` format.  Falls back
                to ``0.0.0.0:8765`` when the URI cannot be parsed.

        Returns:
            A new :class:`GoServer` instance (not yet started).
        """
        import re

        m = re.match(r"ws://([^:/]+):(\d+)", uri)
        host = m.group(1) if m else "0.0.0.0"
        port = int(m.group(2)) if m else 8765
        return cls(host=host, port=port)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the WebSocket server in a background daemon thread.

        This method is idempotent — calling it when the server is already
        running has no effect.  The method blocks until the server socket
        is bound and listening so that callers can immediately call
        :meth:`wait_for_client` afterwards.
        """
        if self._thread is not None and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="GoServer-loop",
        )
        self._thread.start()
        self._server_ready.wait()

    def wait_for_client(self, timeout: float | None = None) -> bool:
        """Block until a Bitburner client connects (or *timeout* expires).

        Args:
            timeout: Maximum number of seconds to wait.  ``None`` means
                wait indefinitely.

        Returns:
            ``True`` if a client is now connected, ``False`` if the wait
            timed out.
        """
        return self._client_connected.wait(timeout=timeout)

    @property
    def is_connected(self) -> bool:
        """``True`` while a Bitburner client has an active connection."""
        return self._client_connected.is_set()

    # ------------------------------------------------------------------
    # Background event-loop helpers
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Entry point for the background thread.

        The event loop is created here so it is owned exclusively by
        this thread, avoiding cross-thread loop sharing.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        """Start the WebSocket server and run until the loop is stopped."""
        async with websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
        ):
            self._server_ready.set()
            # Keep the server alive indefinitely (daemon thread exits
            # when the main process exits).
            await asyncio.Future()

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """Handle a single client connection.

        Stores the active WebSocket and signals :attr:`_client_connected`.
        When the client disconnects the event is cleared so that callers
        waiting on :meth:`wait_for_client` can detect the loss.

        Args:
            ws: The newly accepted server-side WebSocket connection.
        """
        self._ws = ws
        self._client_connected.set()
        try:
            await ws.wait_closed()
        finally:
            # Guard against a second connection having replaced _ws.
            if self._ws is ws:
                self._ws = None
                self._client_connected.clear()

    # ------------------------------------------------------------------
    # Internal send/receive helper
    # ------------------------------------------------------------------

    async def _send_recv(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send *payload* over the active connection and await one reply.

        Args:
            payload: JSON-serialisable mapping to send to the client.

        Returns:
            Parsed JSON response from the client.

        Raises:
            ConnectionError: If no client is currently connected, or if
                the connection drops while waiting for the response.
        """
        if self._ws is None:
            raise ConnectionError(
                "No Bitburner client is connected to the GoServer."
            )
        try:
            await self._ws.send(json.dumps(payload))
            raw = await self._ws.recv()
        except websockets.exceptions.ConnectionClosed as exc:
            raise ConnectionError(
                "Bitburner client disconnected during communication."
            ) from exc
        return json.loads(raw)  # type: ignore[return-value]

    def _run_coro(self, coro: Any) -> Any:
        """Submit *coro* to the background event loop and return its result.

        This is the bridge between the synchronous caller and the async
        WebSocket layer.

        Args:
            coro: Awaitable coroutine to execute on the background loop.

        Returns:
            The return value of *coro*.

        Raises:
            RuntimeError: If :meth:`start` has not been called yet.
        """
        if self._loop is None:
            raise RuntimeError(
                "GoServer event loop is not running.  Call start() first."
            )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ------------------------------------------------------------------
    # Public game API
    # ------------------------------------------------------------------

    def reset(
        self,
        opponent: str = "Netburners",
        board_size: int = 9,
    ) -> dict[str, Any]:
        """Ask the Bitburner client to start a new game and return the state.

        Args:
            opponent: Name of the built-in BitBurner opponent to play
                against (e.g. ``"Netburners"``, ``"Illuminati"``).
            board_size: Side length of the square board (e.g. 5, 7,
                9, or 13).

        Returns:
            Client response dict with keys ``board``, ``current_player``,
            and ``legal_moves``.

        Raises:
            ConnectionError: If the Bitburner client is not connected.
        """
        return self._run_coro(  # type: ignore[return-value]
            self._send_recv(
                {
                    "type": "reset",
                    "opponent": opponent,
                    "board_size": board_size,
                }
            )
        )

    def step(self, action: int) -> dict[str, Any]:
        """Send *action* to the Bitburner client and return the new state.

        Args:
            action: Integer action index.  Values in ``[0, board_size**2)``
                place a stone; the last index (``board_size**2``) is PASS.

        Returns:
            Client response dict with keys ``board``, ``reward``, ``done``,
            ``current_player``, and ``legal_moves``.

        Raises:
            ConnectionError: If the Bitburner client is not connected.
        """
        return self._run_coro(  # type: ignore[return-value]
            self._send_recv({"type": "step", "action": action})
        )
