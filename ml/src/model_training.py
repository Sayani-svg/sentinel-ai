"""The classifier contract, and the boundary a trained model will cross.

:func:`app.services.prediction_service.detect_attack` decides a classification
from a file, not from a row: it aggregates the numeric features into a
:class:`~app.services.prediction_service.FeatureSummary` and applies thresholds
to that. A trained classifier scores each row and aggregates afterwards. The
contract below is deliberately at *file* level so both fit behind it unchanged,
because a boundary drawn at row level would force the rule-based detector to be
rewritten in order to be swappable -- which is the opposite of swappable.

:class:`RuleBasedClassifier` is the current detector behind this contract. It is
not a stub or a placeholder: it is the behaviour the application has today, and
it stays the fallback in :mod:`src.inference` for as long as no artifact exists.

:class:`TrainingRequest` and :class:`TrainingRun` describe what a training run is
and what it must leave behind. No run is performed here. There is no benchmark in
``ml/dataset`` and no scientific Python stack installed, and a training function
that manufactured its own inputs would report an accuracy describing the
manufacture. :func:`train_classifier` says so in a typed error instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Final

from app.schemas.prediction import AttackType
from app.services.prediction_service import (
    AttackDetection,
    DetectionInput,
    detect_attack,
)

from src.config import MODEL_VERSION, RANDOM_STATE
from src.dataset_validation import TrainingDataset
from src.feature_engineering import feature_names

#: Confidence below which a trained model is not trusted to answer alone, and the
#: rule-based detector decides instead. Stated once so the threshold can be tuned
#: without hunting through the selection logic.
MODEL_CONFIDENCE_FLOOR: Final[float] = 0.5


class ClassifierSource(StrEnum):
    """Which classifier produced a result.

    Recorded on every classification so a later review can tell a learned answer
    from a heuristic one. Without it, a report that says ``BruteForce`` gives no
    indication of whether a model or a threshold produced it, which is the first
    question anyone asks of a detection they intend to act on.
    """

    #: A trained artifact decided.
    MODEL = "model"

    #: The rule-based detector decided.
    RULES = "rules"


class TrainingError(RuntimeError):
    """Base class for failures around producing a trained model."""


class TrainingDataUnavailableError(TrainingError):
    """Raised when a training run is attempted with no real dataset.

    Refused rather than satisfied with generated rows. A model trained on
    synthetic data reports a real accuracy figure that describes the synthetic
    data, and that figure then circulates as though it described the threat
    traffic the model will actually see.
    """


class EstimatorUnavailableError(TrainingError):
    """Raised when no estimator implementation is wired up yet."""


@dataclass(frozen=True, slots=True)
class ClassifiedResult:
    """One file's classification, with its provenance.

    Attributes:
        attack_type: The class decided.
        confidence: Strength of the decision, in ``0.0..1.0``.
        source: Which classifier decided.
        detail: Free-form note for logs, such as the rules that fired. Never
            returned to a client.
    """

    attack_type: AttackType
    confidence: float
    source: ClassifierSource
    detail: str = ""


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    """Everything a training run needs stated up front.

    Args are validated by :meth:`validate` rather than at construction, so an
    incomplete request can be built and inspected.

    Attributes:
        dataset: The validated rows to learn from.
        model_name: Identifier of the estimator to fit, matching
            ``Settings.DEFAULT_MODEL``.
        feature_order: Canonical feature order the run trains against. Defaults to
            this build's order.
        random_state: Seed for the split and the estimator, matching
            ``Settings.RANDOM_STATE``.
        version: Version label for the resulting artifact.
    """

    dataset: TrainingDataset
    model_name: str
    feature_order: tuple[str, ...] = field(default_factory=feature_names)
    random_state: int = RANDOM_STATE
    version: str = MODEL_VERSION

    def validate(self) -> None:
        """Check the request can produce a model.

        Raises:
            TrainingError: If the dataset is empty, names no usable feature, or
                the feature order disagrees with the dataset's own.
        """
        if not len(self.dataset):
            raise TrainingDataUnavailableError("The training request holds no rows.")
        if self.dataset.feature_order != self.feature_order:
            raise TrainingError(
                f"The request trains against {list(self.feature_order)} but the "
                f"dataset was validated against "
                f"{list(self.dataset.feature_order)}. They must agree, or the "
                "artifact would be served against vectors it never saw."
            )


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """What a completed training run left behind.

    This is the metadata an artifact has to carry, not a performance claim. The
    ``model_versions`` table holds accuracy and friends, and nothing here is
    written to the database: the models are not this package's to persist.

    Attributes:
        model_name: Estimator that was fitted.
        version: Version label for the artifact.
        feature_order: Feature order the artifact expects, positionally.
        classes: Classes the artifact can predict, in code order.
        rows_trained: Rows the run consumed.
        class_counts: Rows per class at training time.
        trained_at: When the run finished.
    """

    model_name: str
    version: str
    feature_order: tuple[str, ...]
    classes: tuple[AttackType, ...]
    rows_trained: int
    class_counts: dict[AttackType, int]
    trained_at: datetime


class Classifier(ABC):
    """What anything that classifies a log file must be able to do.

    Implementations are interchangeable behind this contract. The application
    depends on the contract, never on a concrete classifier, which is what makes
    the rule-based detector a genuine fallback rather than a branch that has to
    be remembered at every call site.
    """

    #: Feature order this classifier's vectors are positional against.
    @property
    @abstractmethod
    def feature_order(self) -> tuple[str, ...]:
        """Return the feature names, in the order vectors must supply them.

        Returns:
            tuple[str, ...]: One name per position.
        """

    @property
    @abstractmethod
    def source(self) -> ClassifierSource:
        """Return which source this classifier reports.

        Returns:
            ClassifierSource: The value stamped on its results.
        """

    @abstractmethod
    def classify(self, detection_input: DetectionInput) -> ClassifiedResult:
        """Classify one prepared log file.

        Args:
            detection_input: The file's records in the canonical feature space.

        Returns:
            ClassifiedResult: The class, the confidence behind it, and which
            classifier decided.
        """


class RuleBasedClassifier(Classifier):
    """The detector that exists today, behind the shared contract.

    Wraps :func:`~app.services.prediction_service.detect_attack` without
    reinterpreting it. Its thresholds, its corroboration counting and its
    treatment of file-supplied labels are unchanged; the only thing added is the
    provenance stamp, so a result produced by this path is distinguishable from a
    learned one.
    """

    @property
    def feature_order(self) -> tuple[str, ...]:
        """Return the canonical feature order the rules read.

        Returns:
            tuple[str, ...]: One name per position.
        """
        return feature_names()

    @property
    def source(self) -> ClassifierSource:
        """Return that this classifier reports rule-based results.

        Returns:
            ClassifierSource: :attr:`ClassifierSource.RULES`.
        """
        return ClassifierSource.RULES

    def classify(self, detection_input: DetectionInput) -> ClassifiedResult:
        """Classify a file with the existing heuristics.

        Args:
            detection_input: The file's records in the canonical feature space.

        Returns:
            ClassifiedResult: The heuristic verdict, stamped as
            :attr:`ClassifierSource.RULES`.
        """
        detection: AttackDetection = detect_attack(detection_input)
        return ClassifiedResult(
            attack_type=detection.attack_type,
            confidence=detection.confidence,
            source=ClassifierSource.RULES,
            detail="; ".join(detection.matched_rules),
        )


def build_training_run(request: TrainingRequest, *, trained_at: datetime) -> TrainingRun:
    """Describe the artifact a completed run would leave behind.

    Separated from any actual fitting so the shape of a run's metadata is settled
    and testable while the estimator is still unwritten.

    Args:
        request: The validated request describing the run.
        trained_at: When the run finished.

    Returns:
        TrainingRun: Metadata for the resulting artifact.

    Raises:
        TrainingError: If the request does not validate.
    """
    request.validate()
    return TrainingRun(
        model_name=request.model_name,
        version=request.version,
        feature_order=request.feature_order,
        classes=_declared_classes(request.dataset),
        rows_trained=len(request.dataset),
        class_counts=dict(request.dataset.class_counts),
        trained_at=trained_at,
    )


def _declared_classes(dataset: TrainingDataset) -> tuple[AttackType, ...]:
    """Return the classes present, in integer-code order.

    Args:
        dataset: The validated dataset.

    Returns:
        tuple[AttackType, ...]: Classes ordered by their label code, which is
        what a probabilistic estimator's column order is built from.
    """
    from src.preprocessing import label_to_index

    return tuple(sorted(dataset.labels, key=label_to_index))


def train_classifier(
    request: TrainingRequest, estimator: object | None = None
) -> tuple[Classifier, TrainingRun]:
    """Fit a classifier from a validated dataset.

    Refused in this build, with a typed error, rather than quietly doing nothing.
    Two things are missing and neither is a code change: a real benchmark in
    ``ml/dataset``, and a scientific Python stack to fit an estimator with. The
    request is still validated first, so a caller learns immediately whether its
    *data* is usable even before an estimator exists.

    Args:
        request: The run to perform.
        estimator: A fitted-or-fittable estimator. Unused in this build; present
            so the signature does not have to change when one is wired up.

    Returns:
        tuple[Classifier, TrainingRun]: The fitted classifier and its metadata.

    Raises:
        TrainingError: If the request does not validate, no real dataset was
            supplied, or no estimator implementation is available.
    """
    request.validate()

    if not len(request.dataset):
        raise TrainingDataUnavailableError(
            f"The dataset holds no rows. Training needs a real benchmark in the "
            f"dataset directory; this package will not generate one."
        )

    raise EstimatorUnavailableError(
        f"No estimator is wired up for {request.model_name!r}. Install the "
        "scientific Python stack and provide a real dataset before training. "
        "The application continues to classify with the rule-based detector, so "
        "no feature is lost by leaving this unimplemented."
    )


def summarise_run(run: TrainingRun) -> str:
    """Summarise a training run for a log line.

    Args:
        run: The run to describe.

    Returns:
        str: A one-line summary naming the model, version, rows and classes.
    """
    counts = ", ".join(
        f"{attack_type.value}={count}" for attack_type, count in run.class_counts.items()
    )
    return (
        f"{run.model_name} v{run.version}: {run.rows_trained} row(s) over "
        f"{len(run.feature_order)} feature(s); {counts}"
    )


def assert_decision_range(confidence: float) -> float:
    """Check a confidence is in ``0.0..1.0``.

    Args:
        confidence: The confidence to check.

    Returns:
        float: The value, unchanged.

    Raises:
        ValueError: If it falls outside the range the ``predictions.confidence``
            column accepts.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"Confidence {confidence!r} is outside 0.0..1.0, which is the range "
            "the predictions table accepts."
        )
    return confidence