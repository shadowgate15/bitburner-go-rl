"""PPO training pipeline for the TorchRL Go environment."""

from src.train.model import GoActorNet, GoCNN, GoValueNet
from src.train.train import (
    CurriculumTrainConfig,
    TrainConfig,
    _parse_args,
    build_network,
    make_env,
    train,
    train_with_curriculum,
)

__all__ = [
    "CurriculumTrainConfig",
    "GoActorNet",
    "GoCNN",
    "GoValueNet",
    "TrainConfig",
    "_parse_args",
    "build_network",
    "make_env",
    "train",
    "train_with_curriculum",
]
