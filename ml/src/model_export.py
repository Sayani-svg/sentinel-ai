"""Joblib persistence for validated model artifacts.

A fitted estimator is not deployable on its own: the feature order and class
order it was trained against have to travel with it, and both are checked again
when the artifact is read back. Writes go through a temporary file in the model
directory and are published with ``os.replace``, so a failed dump cannot leave a
partially written ``best_model.pkl`` behind.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib

from src.config import MODEL_PATH
from src.feature_engineering import FeatureOrderError
from src.inference import (
    ArtifactContractError,
    ArtifactMetadata,
    ModelArtifact,
    ModelArtifactUnavailableError,
)

ARTIFACT_FORMAT_VERSION: int = 1


def save_artifact(artifact: ModelArtifact, path: Path | None = None) -> Path:
    """Serialize a validated artifact atomically and return the written path."""
    artifact.validate()
    resolved = MODEL_PATH if path is None else path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact": artifact,
        "metadata": artifact.metadata,
    }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            joblib.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return resolved


def _validated_artifact(payload: object, path: Path) -> ModelArtifact:
    """Check a deserialized payload against this build's artifact contract."""
    if not isinstance(payload, dict):
        raise ModelArtifactUnavailableError(
            path, "the file does not contain a model artifact."
        )
    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ModelArtifactUnavailableError(
            path,
            f"unsupported artifact format version {payload.get('format_version')!r}.",
        )

    artifact = payload.get("artifact")
    metadata = payload.get("metadata")
    if not isinstance(artifact, ModelArtifact) or not isinstance(metadata, ArtifactMetadata):
        raise ModelArtifactUnavailableError(
            path, "the artifact payload is missing its estimator or metadata."
        )
    if artifact.metadata != metadata:
        raise ArtifactContractError(
            "The artifact's recorded metadata disagrees with the metadata stored "
            "beside it."
        )
    try:
        artifact.validate()
    except (ArtifactContractError, FeatureOrderError) as exc:
        raise ArtifactContractError(
            f"The serialized artifact failed validation: {exc}"
        ) from exc
    return artifact


def load_artifact(path: Path | None = None) -> ModelArtifact:
    """Read and validate a joblib artifact, defaulting to ``MODEL_PATH``."""
    resolved = MODEL_PATH if path is None else path

    if not resolved.exists():
        raise ModelArtifactUnavailableError(resolved, "the file does not exist.")
    if resolved.is_dir():
        raise ModelArtifactUnavailableError(resolved, "the path is a directory.")
    if resolved.stat().st_size == 0:
        raise ModelArtifactUnavailableError(
            resolved,
            "the file is empty, which means a placeholder was written where an "
            "artifact was expected.",
        )

    try:
        payload = joblib.load(resolved)
    except Exception as exc:
        raise ModelArtifactUnavailableError(
            resolved, f"joblib could not read the artifact: {exc}"
        ) from exc
    return _validated_artifact(payload, resolved)


__all__ = ["ARTIFACT_FORMAT_VERSION", "load_artifact", "save_artifact"]
