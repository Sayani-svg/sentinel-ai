"""Evaluation must refuse a partial dataset unless explicitly allowed."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.schemas.prediction import AttackType

from src.dataset_validation import (
    DatasetCompleteness,
    IncompleteDatasetError,
    dataset_from_records,
)
from src.evaluation import EvaluationDataError, assert_evaluation_ready
from src.feature_engineering import feature_names


def _dataset(*, complete: bool) -> "object":
    """Return a two-class dataset with the requested completeness record."""
    width = len(feature_names())
    completeness = DatasetCompleteness(complete=complete)
    if not complete:
        completeness = DatasetCompleteness(
            complete=False,
            truncated_paths=(Path("day1.csv"),),
            reasons=("day1.csv exceeded the backend parser row cap and was truncated.",),
        )
    return dataset_from_records(
        [
            ((1.0,) * width, AttackType.BENIGN),
            ((2.0,) * width, AttackType.DOS),
        ],
        completeness=completeness,
    )


def test_complete_dataset_is_evaluation_ready() -> None:
    """A complete source load needs no opt-in."""
    assert_evaluation_ready(_dataset(complete=True))


def test_incomplete_dataset_is_refused_by_default() -> None:
    """Evaluation must not report metrics over a capped view of the source."""
    with pytest.raises(EvaluationDataError) as excinfo:
        assert_evaluation_ready(_dataset(complete=False))

    assert "allow_incomplete_dataset=True" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, IncompleteDatasetError)


def test_explicit_incomplete_dataset_opt_in_allows_evaluation() -> None:
    """The same named opt-in as training must permit an incomplete evaluation."""
    assert_evaluation_ready(
        _dataset(complete=False), allow_incomplete_dataset=True
    )
