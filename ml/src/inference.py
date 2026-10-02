"""Serving: loading a trained artifact, and falling back to the rules when there isn't one.

Nothing here loads a model today. :func:`load_artifact` refuses, with a typed
error naming the path it looked at, because ``ml/models`` holds only a
``.gitkeep`` and no serializer is installed. An empty ``best_model.pkl`` -- a
zero-byte placeholder, or a pickle of ``None`` -- would be worse than absent,
because its presence would satisfy a presence check and the application would
silently serve nothing.

**The fallback is observable.** :class:`FallbackClassifier` tries the model and
defers to the rules when the model is unavailable or not confident enough, and
every result it returns carries :attr:`~src.model_training.ClassifiedResult.source`
naming which path decided. That stamp is the whole point: a fallback that cannot
be distinguished from a model decision is a fallback nobody can audit, and a
detection nobody can audit is one nobody trusts.

:class:`~src.model_training.Classifier` is file-level because the rule-based
detector aggregates a whole file before deciding, so the model path aggregates
per-row scores the same way. Both fit behind one contract unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.schemas.prediction import AttackType
from app.services.prediction_service import DetectionInput

from src.config import MODEL_PATH
from src.feature_engineering import (
    FeatureOrderError,
    assert_feature_order,
    to_feature_matrix,
)
from src.model_training import (
    MODEL_CONFIDENCE_FLOOR,
    Classifier,
    ClassifierSource,
    ClassifiedResult,
    RuleBasedClassifier,
)
from src.preprocessing import TRAINABLE_ATTACK_TYPES


class InferenceError(RuntimeError):
    """Base class for failures serving a model."""


class ModelArtifactUnavailableError(InferenceError):
    """Raised when no usable artifact exists at the expected path.

    Carries the path so the message can name it. "Model not loaded" without a
    location is the least actionable error in this codebase: nobody can tell
    whether to generate the artifact, fix the path, or give up.
    """

    def __init__(self, path: Path, reason: str) -> None:
        """Record where the artifact was expected and why it could not be used.

        Args:
            path: The path that was looked at.
            reason: Why it is not usable.
        """
        super().__init__(
            f"No usable model artifact at {path}: {reason}. The rule-based "
            "detector continues to serve classifications, so this is a "
            "degradation rather than an outage."
        )
        self.path = path
        self.reason = reason


class SerializationUnavailableError(ModelArtifactUnavailableError):
    """Raised when no serializer is available to read an artifact."""


class ArtifactContractError(InferenceError):
    """Raised when an artifact's recorded metadata does not match this build."""


@runtime_checkable
class ProbabilisticEstimator(Protocol):
    """What this module needs from a fitted estimator.

    Structural on purpose: an estimator from any library satisfies it by having
    these attributes, so the serving path does not import a machine learning
    package to serve a model. :attr:`classes_` holds the attack types in the same
    order as the probability columns ``predict_proba`` returns, which is the one
    convention this module relies on and the one an artifact must record.
    """

    classes_: Sequence[AttackType]

    def predict_proba(self, features: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        """Return per-class probabilities for each row.

        Args:
            features: A row-major matrix, one row per record.

        Returns:
            Sequence[Sequence[float]]: One row of probabilities per input row,
            each aligned with :attr:`classes_`.
        """
        ...


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """What a serialized artifact must record beside the estimator.

    A pickled estimator alone is not deployable. Without :attr:`feature_order`,
    a vector built in a different column order is fed to the model and produces
    confident wrong answers; without :attr:`classes`, its probability columns
    cannot be read back to labels at all. Both are therefore stored with the
    artifact and both are checked on load.

    Attributes:
        model_name: Estimator identifier, matching ``Settings.DEFAULT_MODEL``.
        version: Version label, matching ``Settings.MODEL_VERSION``.
        feature_order: Feature names, in the positional order the estimator expects.
        classes: Classes, in the order ``predict_proba`` returns its columns.
    """

    model_name: str
    version: str
    feature_order: tuple[str, ...]
    classes: tuple[AttackType, ...]

    def validate(self) -> None:
        """Check the metadata against this build's contract.

        Raises:
            ArtifactContractError: If the recorded classes are not the trainable
                vocabulary, or if the probability column count would not match.
            FeatureOrderError: If the feature order does not match this build's.
        """
        unknown = [
            attack_type.value
            for attack_type in self.classes
            if attack_type not in TRAINABLE_ATTACK_TYPES
        ]
        if unknown:
            raise ArtifactContractError(
                f"The artifact predicts {', '.join(unknown)}, which this build "
                f"does not define. Expected classes drawn from: "
                f"{', '.join(a.value for a in TRAINABLE_ATTACK_TYPES)}."
            )
        assert_feature_order(self.feature_order)


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A fitted estimator plus the metadata needed to serve it correctly.

    Attributes:
        estimator: The fitted estimator.
        metadata: Feature order and class vocabulary it was trained against.
    """

    estimator: ProbabilisticEstimator
    metadata: ArtifactMetadata

    def validate(self) -> None:
        """Check the artifact is consistent with itself and with this build.

        Raises:
            ArtifactContractError: If the estimator's class order disagrees with
                the metadata's, which would mislabel every probability column.
            FeatureOrderError: If the recorded feature order is wrong.
        """
        self.metadata.validate()
        estimator_classes = tuple(self.estimator.classes_)
        if estimator_classes != self.metadata.classes:
            raise ArtifactContractError(
                f"The estimator's class order {estimator_classes} disagrees with "
                f"the artifact metadata {self.metadata.classes}. Probability "
                "columns would be read as the wrong labels."
            )


class ModelClassifier(Classifier):
    """A fitted estimator behind the shared classifier contract.

    Scores every record, then decides once for the file. Scoring per row and
    aggregating is the only shape that mirrors what the rule-based detector
    already does, so swapping the two changes which path produced a verdict
    without changing what a verdict covers.
    """

    def __init__(self, artifact: ModelArtifact, *, confidence_floor: float = MODEL_CONFIDENCE_FLOOR) -> None:
        """Wrap an artifact for serving.

        Args:
            artifact: The estimator and its metadata. Validated immediately, so a
                mismatched artifact fails here rather than at the first request.
            confidence_floor: Below this, :class:`FallbackClassifier` defers to
                the rules. Unused when this classifier is used directly.
        """
        artifact.validate()
        self._artifact = artifact
        self._confidence_floor = confidence_floor

    @property
    def artifact(self) -> ModelArtifact:
        """Return the wrapped artifact.

        Returns:
            ModelArtifact: The estimator and metadata being served.
        """
        return self._artifact

    @property
    def confidence_floor(self) -> float:
        """Return the confidence below which the rules take over.

        Returns:
            float: The threshold in ``0.0..1.0``.
        """
        return self._confidence_floor

    @property
    def feature_order(self) -> tuple[str, ...]:
        """Return the feature order the artifact was trained against.

        Returns:
            tuple[str, ...]: One name per position.
        """
        return self._artifact.metadata.feature_order

    @property
    def source(self) -> ClassifierSource:
        """Return that this classifier reports model results.

        Returns:
            ClassifierSource: :attr:`ClassifierSource.MODEL`.
        """
        return ClassifierSource.MODEL

    def score_rows(self, detection_input: DetectionInput) -> tuple[tuple[AttackType, float], ...]:
        """Return a per-record class and confidence.

        Args:
            detection_input: The file's records in the canonical feature space.

        Returns:
            tuple[tuple[AttackType, float], ...]: One ``(class, confidence)`` per
            record, in record order.
        """
        records = detection_input.records
        if not records:
            return ()

        matrix = [list(row) for row in to_feature_matrix(records)]
        classes = tuple(self._artifact.metadata.classes)
        probabilities = self._artifact.estimator.predict_proba(matrix)

        scores: list[tuple[AttackType, float]] = []
        for row in probabilities:
            best = max(range(len(row)), key=lambda index: (row[index], -index))
            scores.append((classes[best], float(row[best])))
        return tuple(scores)

    def classify(self, detection_input: DetectionInput) -> ClassifiedResult:
        """Classify a file from its per-record model scores.

        The winning class is the one carrying the greatest total probability mass,
        with the class code as a deterministic tie-break so repeated runs on
        unchanged data agree. Confidence is that mass as a fraction of the record
        count, which stays in ``0.0..1.0`` by construction.

        Args:
            detection_input: The file's records in the canonical feature space.

        Returns:
            ClassifiedResult: The aggregated verdict, stamped as
            :attr:`ClassifierSource.MODEL`.

        Raises:
            InferenceError: If the file carried no records, or the estimator
                returned a column count that does not match its classes.
        """
        scores = self.score_rows(detection_input)
        if not scores:
            raise InferenceError(
                "The file carried no records to classify. Upload validation "
                "should have refused it before it reached a classifier."
            )

        mass: dict[AttackType, float] = {}
        for attack_type, confidence in scores:
            mass[attack_type] = mass.get(attack_type, 0.0) + confidence

        classes = tuple(self._artifact.metadata.classes)
        winner = min(
            mass, key=lambda attack_type: (-mass[attack_type], classes.index(attack_type))
        )
        confidence = mass[winner] / len(scores)

        return ClassifiedResult(
            attack_type=winner,
            confidence=confidence,
            source=ClassifierSource.MODEL,
            detail=(
                f"{self._artifact.metadata.model_name} "
                f"v{self._artifact.metadata.version} over {len(scores)} record(s)"
            ),
        )


class FallbackClassifier(Classifier):
    """The model path, with the rule-based detector behind it.

    Delegates to :attr:`fallback` when the model is absent, or when the model's
    confidence is below its floor. Either way the result says which path decided,
    so a deployment running on thresholds is visible in its own output rather
    than discovered during an incident review.
    """

    def __init__(
        self,
        model: Classifier | None,
        fallback: Classifier,
        *,
        confidence_floor: float = MODEL_CONFIDENCE_FLOOR,
    ) -> None:
        """Compose a model path with a fallback.

        Args:
            model: The trained path, or ``None`` when no artifact exists.
            fallback: The path to defer to. Defaults are not assumed here; the
                caller passes :class:`~src.model_training.RuleBasedClassifier`.
            confidence_floor: Model confidence below which the fallback decides.
        """
        self._model = model
        self._fallback = fallback
        self._confidence_floor = confidence_floor

    @property
    def feature_order(self) -> tuple[str, ...]:
        """Return the model's feature order when one is loaded.

        Returns:
            tuple[str, ...]: The model's order, else the fallback's.
        """
        source = self._model if self._model is not None else self._fallback
        return source.feature_order

    @property
    def source(self) -> ClassifierSource:
        """Return that results from this classifier may be either source.

        Returns:
            ClassifierSource: The model source when one is loaded. The value on a
            *result* is authoritative; this property describes the preferred path.
        """
        if self._model is None:
            return ClassifierSource.RULES
        return ClassifierSource.MODEL

    @property
    def has_model(self) -> bool:
        """Return whether a model path is loaded.

        Returns:
            bool: ``True`` when a model is available to try.
        """
        return self._model is not None

    def classify(self, detection_input: DetectionInput) -> ClassifiedResult:
        """Classify a file, preferring the model and deferring when it is unsure.

        Args:
            detection_input: The file's records in the canonical feature space.

        Returns:
            ClassifiedResult: The model's verdict when it is confident enough,
            otherwise the fallback's, with :attr:`ClassifiedResult.source` naming
            which path produced it.
        """
        if self._model is None:
            return self._fallback.classify(detection_input)

        result = self._model.classify(detection_input)
        if result.confidence >= self._confidence_floor:
            return result

        deferred = self._fallback.classify(detection_input)
        return ClassifiedResult(
            attack_type=deferred.attack_type,
            confidence=deferred.confidence,
            source=ClassifierSource.RULES,
            detail=(
                f"model confidence {result.confidence:.3f} below floor "
                f"{self._confidence_floor:.3f}; deferred to rules "
                f"(model said {result.attack_type.value})"
            ),
        )


def load_artifact(path: Path | None = None) -> ModelArtifact:
    """Load a serialized artifact from disk.

    Refused in this build. No artifact exists and no serializer is installed, so
    rather than returning ``None`` -- which a caller could mistake for "no model
    available, use the rules" and skip -- this raises a typed error naming the
    path. :func:`select_classifier` is the entry point that turns absence into a
    fallback instead of an exception.

    Args:
        path: The artifact to read, or ``None`` for ``Settings.MODEL_PATH``.

    Returns:
        ModelArtifact: The loaded artifact.

    Raises:
        ModelArtifactUnavailableError: If the file is absent or empty.
        SerializationUnavailableError: If a file is present but no serializer is
            available to read it.
    """
    resolved = MODEL_PATH if path is None else path

    if not resolved.exists():
        raise ModelArtifactUnavailableError(
            resolved, "the file does not exist."
        )
    if resolved.is_dir():
        raise ModelArtifactUnavailableError(resolved, "the path is a directory.")
    if resolved.stat().st_size == 0:
        raise ModelArtifactUnavailableError(
            resolved,
            "the file is empty, which means a placeholder was written where an "
            "artifact was expected.",
        )

    raise SerializationUnavailableError(
        resolved,
        "reading an artifact needs a serializer such as joblib or pickle, and "
        "none is installed. The scientific Python stack must be installed "
        "before a trained model can be served.",
    )


def select_classifier(
    path: Path | None = None,
    *,
    fallback: Classifier | None = None,
    confidence_floor: float = MODEL_CONFIDENCE_FLOOR,
) -> FallbackClassifier:
    """Return the classifier to serve with, falling back when no model exists.

    This is the single place absence becomes a fallback. Callers get a usable
    classifier and never handle an artifact error, which is what keeps the
    fallback from being forgotten at a call site.

    Args:
        path: The artifact to try, or ``None`` for ``Settings.MODEL_PATH``.
        fallback: The rules to defer to, or ``None`` for a
            :class:`~src.model_training.RuleBasedClassifier`.
        confidence_floor: Model confidence below which the fallback decides.

    Returns:
        FallbackClassifier: A classifier that serves the model when it can and
        the rules when it cannot.
    """
    rules = RuleBasedClassifier() if fallback is None else fallback

    try:
        artifact = load_artifact(path)
    except InferenceError:
        return FallbackClassifier(None, rules, confidence_floor=confidence_floor)

    return FallbackClassifier(
        ModelClassifier(artifact, confidence_floor=confidence_floor),
        rules,
        confidence_floor=confidence_floor,
    )


__all__ = [
    "ArtifactContractError",
    "ArtifactMetadata",
    "FeatureOrderError",
    "FallbackClassifier",
    "InferenceError",
    "ModelArtifact",
    "ModelArtifactUnavailableError",
    "ModelClassifier",
    "ProbabilisticEstimator",
    "SerializationUnavailableError",
    "load_artifact",
    "select_classifier",
]