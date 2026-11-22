"""Artifact persistence must round-trip the fitted contract, not just the estimator."""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pytest
from app.services.log_parser import parse_csv_log

from src import model_export
from src.config import RANDOM_FOREST_MODEL
from src.dataset_validation import dataset_from_records
from src.feature_engineering import feature_names, to_feature_space
from src.inference import (
    ArtifactContractError,
    ModelArtifactUnavailableError,
    ModelClassifier,
    load_artifact,
    select_classifier,
)
from src.model_export import ARTIFACT_FORMAT_VERSION, save_artifact
from src.model_training import ClassifierSource, TrainingRequest, train_classifier
from src.preprocessing import CLASS_COUNT, TRAINABLE_ATTACK_TYPES


def _dataset():
    """Return a two-class in-memory dataset for a real scikit-learn fit."""
    width = len(feature_names())
    from app.schemas.prediction import AttackType

    return dataset_from_records(
        [
            ((1.0,) * width, AttackType.BENIGN),
            ((2.0,) * width, AttackType.DOS),
        ]
    )


def _artifact():
    """Return a trained artifact whose persistence is under test."""
    classifier, _ = train_classifier(
        TrainingRequest(dataset=_dataset(), model_name=RANDOM_FOREST_MODEL)
    )
    return classifier.artifact


def _detection_input():
    """Return a prepared file for the serving half of the round trip."""
    return to_feature_space(
        parse_csv_log(
            io.StringIO(
                "Flow Duration,Total Length,Total Packets,Avg Packet Size,Packets Per Second\n"
                "10,1000,5,200,1\n20,2000,10,200,2\n"
            )
        )
    )


def test_random_forest_artifact_round_trip_classifies(tmp_path: Path) -> None:
    """A saved forest must reload with its full contract and serve predictions."""
    original = _artifact()
    target = tmp_path / "best_model.pkl"

    written = save_artifact(original, target)
    loaded = load_artifact(written)

    assert written == target
    assert loaded.metadata == original.metadata
    assert loaded.metadata.feature_order == feature_names()
    assert loaded.metadata.classes == TRAINABLE_ATTACK_TYPES
    assert loaded.estimator.classes_ == TRAINABLE_ATTACK_TYPES

    rows = [row.features for row in _dataset().examples]
    probabilities = loaded.estimator.predict_proba(rows)
    assert probabilities.shape == (len(rows), CLASS_COUNT)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(len(rows)))

    result = ModelClassifier(loaded).classify(_detection_input())
    assert result.source is ClassifierSource.MODEL
    assert result.attack_type in TRAINABLE_ATTACK_TYPES
    assert select_classifier(target).has_model


def test_successful_save_leaves_no_temporary_files(tmp_path: Path) -> None:
    """A completed write must publish only the final artifact name."""
    target = tmp_path / "best_model.pkl"

    save_artifact(_artifact(), target)

    assert target.exists()
    assert list(tmp_path.glob(".*best_model.pkl.*")) == []


def test_atomic_save_removes_partial_files_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed dump must leave neither a final artifact nor a temporary file."""

    def failing_dump(payload: object, handle: object) -> None:
        handle.write(b"partial")
        raise RuntimeError("dump failed")

    monkeypatch.setattr(model_export.joblib, "dump", failing_dump)
    target = tmp_path / "best_model.pkl"

    with pytest.raises(RuntimeError, match="dump failed"):
        save_artifact(_artifact(), target)

    assert not target.exists()
    assert list(tmp_path.glob(".*best_model.pkl.*")) == []


def test_load_rejects_tampered_metadata(tmp_path: Path) -> None:
    """A valid estimator with a broken feature order must not reach serving."""
    artifact = _artifact()
    target = tmp_path / "tampered.pkl"
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact": artifact,
        "metadata": replace(
            artifact.metadata, feature_order=tuple(reversed(feature_names()))
        ),
    }
    joblib.dump(payload, target)

    with pytest.raises(ArtifactContractError):
        load_artifact(target)


def test_load_rejects_unknown_format_version(tmp_path: Path) -> None:
    """A future or corrupt payload must be refused as an unavailable artifact."""
    target = tmp_path / "future.pkl"
    joblib.dump({"format_version": ARTIFACT_FORMAT_VERSION + 1}, target)

    with pytest.raises(ModelArtifactUnavailableError):
        load_artifact(target)
