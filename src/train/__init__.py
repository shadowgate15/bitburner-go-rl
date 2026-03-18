"""PPO training pipeline for the TorchRL Go environment."""

from src.train.model import GoActorNet, GoCNN, GoValueNet
from src.train.train import (
    TrainConfig,
    _parse_args,
    _parse_args_curriculum,
    build_network,
    make_env,
    train,
    train_with_curriculum,
)

__all__ = [
    "GoActorNet",
    "GoCNN",
    "GoValueNet",
    "TrainConfig",
    "_parse_args",
    "_parse_args_curriculum",
    "build_network",
    "make_env",
    "train",
    "train_with_curriculum",
]
