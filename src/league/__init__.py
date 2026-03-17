"""Curriculum learning and league self-play system for IPvGO.

Public exports
--------------
* :class:`~src.league.curriculum.BoardCurriculum`
* :class:`~src.league.opponents.BuiltinOpponent`
* :class:`~src.league.opponents.ModelOpponent`
* :class:`~src.league.opponents.OpponentPool`
* :class:`~src.league.opponents.OpponentProtocol`
* :class:`~src.league.opponents.RandomOpponent`
* :class:`~src.league.checkpoint.CheckpointManager`
* :func:`~src.league.rollout.play_episode`
* :func:`~src.league.evaluation.evaluate`
"""

from src.league.checkpoint import CheckpointManager
from src.league.curriculum import BoardCurriculum
from src.league.evaluation import evaluate
from src.league.opponents import (
    BuiltinOpponent,
    ModelOpponent,
    OpponentPool,
    OpponentProtocol,
    RandomOpponent,
)
from src.league.rollout import play_episode

__all__ = [
    "BoardCurriculum",
    "BuiltinOpponent",
    "CheckpointManager",
    "ModelOpponent",
    "OpponentPool",
    "OpponentProtocol",
    "RandomOpponent",
    "evaluate",
    "play_episode",
]
