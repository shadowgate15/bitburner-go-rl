"""Single-episode rollout for self-play and evaluation.

:func:`play_episode` drives one complete game between a learning
agent and any opponent drawn from the league pool.  Only the learning
agent's perspective is used (the agent is always the *active* player
from the environment's point of view).

The function does **not** accumulate experience for PPO updates;
it exists solely to produce win/loss signals used by the evaluation
system and curriculum.

IPvGO WebSocket usage
---------------------
**All** opponent types use the IPvGO WebSocket for game-state
management, but in different ways:

* :class:`~src.league.opponents.BuiltinOpponent` - a single
  :meth:`~src.env.client.GoClient.builtin_step` call both selects the
  bot's move and advances game state server-side.  The caller must
  **not** call :meth:`~src.env.go_env.TorchRLGoEnv._step` afterwards.
  :func:`play_episode` handles this automatically: it uses
  :meth:`~src.env.go_env.TorchRLGoEnv._encode_step_response` to
  convert the server's response into a TensorDict without sending
  another message.

* :class:`~src.league.opponents.ModelOpponent` - move selection is
  **local** (frozen neural network inference); game-state advancement
  goes through :meth:`~src.env.go_env.TorchRLGoEnv._step`.

* :class:`~src.league.opponents.RandomOpponent` - move selection is
  **local** (uniform sample from legal-move mask); game-state
  advancement goes through :meth:`~src.env.go_env.TorchRLGoEnv._step`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from src.league.opponents import BuiltinOpponent

if TYPE_CHECKING:  # pragma: no cover
    from src.env.go_env import TorchRLGoEnv
    from src.league.opponents import OpponentProtocol


def play_episode(
    env: TorchRLGoEnv,
    agent_actor: Any,
    opponent: BuiltinOpponent | OpponentProtocol,
    board_size: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Play one complete game and return episode statistics.

    Turn order:

    1. **Agent** - called via the TorchRL actor; game-state advancement
       goes through :meth:`~src.env.go_env.TorchRLGoEnv._step`.
    2. **Opponent** - handling depends on the opponent type:

       * :class:`~src.league.opponents.BuiltinOpponent`: calls
         :meth:`~src.league.opponents.BuiltinOpponent.step`, which
         drives the server bot to play and advances state in one
         WebSocket round-trip.  No :meth:`_step` call is made.
       * All other opponents (implementing
         :class:`~src.league.opponents.OpponentProtocol`): calls
         ``opponent.act(obs)`` to get an action, then advances state
         via :meth:`~src.env.go_env.TorchRLGoEnv._step`.

    The two opponent interfaces are intentionally different:
    :class:`~src.league.opponents.BuiltinOpponent` advances game state
    server-side as part of move selection, whereas all other opponents
    select a move locally and delegate state advancement to the
    environment.

    The episode terminates when either step returns ``done=True``.

    Only the agent's accumulated reward is used to determine the
    win/loss result.  The opponent receives no gradient signal (its
    forward pass is guarded by :func:`torch.no_grad`).

    Args:
        env: A freshly-created (or recycled) :class:`TorchRLGoEnv`
            instance.  The caller is responsible for ensuring that the
            environment matches *board_size*.
        agent_actor: Trained TorchRL actor (``ProbabilisticActor`` or
            equivalent).  Called with a batched ``TensorDict`` whose
            ``"observation"`` key holds shape
            ``(1, 4, board_size, board_size)``.
        opponent: Either a
            :class:`~src.league.opponents.BuiltinOpponent` (server-
            driven, ``step()`` interface) or any object satisfying
            :class:`~src.league.opponents.OpponentProtocol` (local
            move selection, ``act()`` interface).
        board_size: Side length of the board (used for action
            bookkeeping only; the environment spec is not re-checked).
        device: Device to run agent inference on.

    Returns:
        Dict with the following keys:

        * ``"won"``          - ``True`` if the agent's total reward > 0.
        * ``"total_reward"`` - Sum of all rewards received by the agent.
        * ``"steps"``        - Number of agent steps taken.
    """
    _device = torch.device(device)

    # ---- reset -------------------------------------------------------
    reset_td = env._reset()
    obs: torch.Tensor = reset_td["observation"]  # (4, B, B)
    done: bool = bool(reset_td.get("done", torch.tensor([False]))[0].item())

    total_reward = 0.0
    steps = 0

    while not done:
        # ---- agent turn ----------------------------------------------
        with torch.no_grad():
            agent_obs = obs.to(_device).unsqueeze(0)  # (1, 4, B, B)
            td_in = TensorDict(
                {"observation": agent_obs}, batch_size=[1]
            )
            td_out = agent_actor(td_in)
            agent_action: int = int(td_out["action"][0].item())

        step_td = env._step(
            TensorDict(
                {"action": torch.tensor(agent_action, dtype=torch.int64)},
                batch_size=[],
            )
        )

        obs = step_td["observation"]
        total_reward += float(step_td["reward"].item())
        done = bool(step_td["done"].item())
        steps += 1

        if done:
            break

        # ---- opponent turn -------------------------------------------
        if isinstance(opponent, BuiltinOpponent):
            # The builtin bot selects and plays its move server-side in
            # one WebSocket round-trip.  Game state is already advanced
            # when step() returns; do NOT call env._step() afterwards.
            opp_response = opponent.step()
            opp_td = env._encode_step_response(opp_response)
        else:
            with torch.no_grad():
                opp_action: int = opponent.act(obs)  # type: ignore[union-attr]

            opp_td = env._step(
                TensorDict(
                    {"action": torch.tensor(opp_action, dtype=torch.int64)},
                    batch_size=[],
                )
            )

        obs = opp_td["observation"]
        # Capture the terminal reward even when the episode ends on
        # the opponent's turn (it still represents the agent's outcome).
        total_reward += float(opp_td["reward"].item())
        done = bool(opp_td["done"].item())

    return {
        "won": total_reward > 0.0,
        "total_reward": total_reward,
        "steps": steps,
    }


__all__ = ["play_episode"]
