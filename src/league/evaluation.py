"""Evaluation helpers for the curriculum / league training system.

:func:`evaluate` runs a fixed number of games against each opponent
category and returns a structured metrics dict.  The metrics drive
both :class:`~src.league.curriculum.BoardCurriculum` advancement and
:class:`~src.league.checkpoint.CheckpointManager` best-model
selection.

Opponent categories evaluated:

* Built-in bots: ``easy``, ``medium``, ``hard``
* League - latest in-memory actor (or historical if unavailable)
* Random baseline

Return structure::

    {
        "win_rate_vs_easy":    float,  # [0, 1]
        "win_rate_vs_medium":  float,
        "win_rate_vs_hard":    float,
        "win_rate_vs_league":  float,
        "win_rate_vs_random":  float,
        "avg_reward":          float,
    }
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from src.league.opponents import ModelOpponent, RandomOpponent
from src.league.rollout import play_episode

if TYPE_CHECKING:  # pragma: no cover
    from src.env.go_env import TorchRLGoEnv
    from src.league.opponents import OpponentPool


def evaluate(
    agent_actor: Any,
    opponent_pool: OpponentPool,
    env_factory: Callable[[], TorchRLGoEnv],
    board_size: int,
    num_games: int = 50,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Evaluate the agent against all opponent categories.

    Runs *num_games* episodes against each opponent type in sequence.
    For built-in bots the game result depends on the server's
    implementation of
    :meth:`~src.env.client.GoClient.get_builtin_move`; when that API
    is not yet available an :exc:`NotImplementedError` will be raised.

    Args:
        agent_actor: The current learning policy (TorchRL actor).
        opponent_pool: :class:`~src.league.opponents.OpponentPool`
            owning built-in and historical opponents.
        env_factory: Zero-argument callable that returns a fresh
            :class:`~src.env.go_env.TorchRLGoEnv` for each game.
        board_size: Side length of the board - used when constructing
            the random baseline opponent.
        num_games: Number of games to play per opponent (default 50).
        device: Device to run agent inference on.

    Returns:
        Dict with keys ``"win_rate_vs_easy"``,
        ``"win_rate_vs_medium"``, ``"win_rate_vs_hard"``,
        ``"win_rate_vs_league"``, ``"win_rate_vs_random"``, and
        ``"avg_reward"``.
    """
    metrics: dict[str, float] = {}
    all_rewards: list[float] = []

    def _run(opponent: Any, n: int) -> float:
        """Run *n* episodes and return the win rate.

        Args:
            opponent: Opponent implementing :class:`OpponentProtocol`.
            n: Number of games.

        Returns:
            Win rate in ``[0, 1]``.
        """
        wins = 0
        for _ in range(n):
            env = env_factory()
            result = play_episode(
                env, agent_actor, opponent, board_size, device
            )
            wins += int(result["won"])
            all_rewards.append(float(result["total_reward"]))
        return wins / n if n > 0 else 0.0

    # ---- built-in bots -----------------------------------------------
    for bot_name in ["easy", "medium", "hard"]:
        opp = opponent_pool.get_builtin_opponent(bot_name)
        metrics[f"win_rate_vs_{bot_name}"] = _run(opp, num_games)

    # ---- league (latest actor or historical sample) ------------------
    if opponent_pool.latest_actor is not None:
        league_opp: Any = ModelOpponent(
            opponent_pool.latest_actor, device=device
        )
    elif opponent_pool.latest_path is not None:
        league_opp = opponent_pool.load_model_opponent(
            opponent_pool.latest_path
        )
    else:
        league_opp = RandomOpponent(board_size * board_size + 1)

    metrics["win_rate_vs_league"] = _run(league_opp, num_games)

    # ---- random baseline ---------------------------------------------
    random_opp = RandomOpponent(board_size * board_size + 1)
    metrics["win_rate_vs_random"] = _run(random_opp, num_games)

    # ---- aggregate reward -------------------------------------------
    metrics["avg_reward"] = (
        sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    )

    return metrics


__all__ = ["evaluate"]
