"""Curriculum learning package for the BitBurner IPvGO RL agent."""

from src.curriculum.curriculum import (
    BOARD_SIZES,
    OPPONENTS,
    CurriculumConfig,
    GoCurriculumManager,
)

__all__ = [
    "BOARD_SIZES",
    "OPPONENTS",
    "CurriculumConfig",
    "GoCurriculumManager",
]
