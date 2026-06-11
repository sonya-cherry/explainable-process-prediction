"""Reusable end-to-end pipelines for process outcome prediction."""

from src.pipeline.training_pipeline import (
    TrainingPipelineResult,
    run_training_pipeline,
)

__all__ = [
    "TrainingPipelineResult",
    "run_training_pipeline",
]
