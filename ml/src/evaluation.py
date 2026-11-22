"""Evaluation-side dataset readiness checks.

There is no metric implementation in this build. The readiness guard is still
part of the evaluation contract: an incomplete source load must be rejected by
default, with the same explicit opt-in used by training, before any evaluator is
wired in.
"""

from __future__ import annotations

from src.dataset_validation import (
    IncompleteDatasetError,
    TrainingDataset,
    require_complete_dataset,
)


class EvaluationDataError(ValueError):
    """Raised when a dataset is not fit for evaluation."""


def assert_evaluation_ready(
    dataset: TrainingDataset, *, allow_incomplete_dataset: bool = False
) -> None:
    """Check that a dataset is complete enough to evaluate.

    Args:
        dataset: Dataset about to be evaluated.
        allow_incomplete_dataset: Explicit opt-in for a partial source load.

    Raises:
        EvaluationDataError: If the dataset was partially loaded without opt-in.
    """
    try:
        require_complete_dataset(
            dataset, allow_incomplete=allow_incomplete_dataset
        )
    except IncompleteDatasetError as exc:
        raise EvaluationDataError(str(exc)) from exc


__all__ = ["EvaluationDataError", "assert_evaluation_ready"]
