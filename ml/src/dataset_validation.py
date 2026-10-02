"""The training dataset contract, and the rules a dataset must satisfy.

This module defines what a training dataset *is* and refuses datasets that cannot
produce an honest model. It deliberately does not create one. There is no data
in ``ml/dataset`` -- the directory holds a ``.gitkeep`` -- and fabricating rows
to demonstrate a training run would produce a model whose reported accuracy
describes the fabrication. The checks below are therefore the deliverable: they
say what will be accepted, so the checks can be run against a real benchmark the
moment one is placed.

A dataset here is a list of :class:`LabeledExample`: one feature vector and the
:class:`~app.schemas.prediction.AttackType` that vector is ground truth for. The
feature ordering is not carried per-example; it is stated once for the whole
dataset, because a vector is positional and a per-row ordering could disagree
with itself without anything noticing.

The validation is strict about three things that quietly ruin a classifier:

* **Completeness.** Every example carries every declared feature. See
  :mod:`src.feature_engineering` for why absence is not imputed.
* **Finiteness.** No ``NaN`` or infinity. These survive most arithmetic and
  produce a model that trains without complaint and predicts nonsensically.
* **Class coverage.** At least two distinct classes. A dataset of one class
  trains a model that always answers that class, which scores perfectly on its
  own training data and is worse than useless on real traffic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.schemas.prediction import AttackType

from src.feature_engineering import feature_names
from src.preprocessing import TRAINABLE_ATTACK_TYPES


class DatasetValidationError(ValueError):
    """Base class for datasets that cannot be trained on."""


class EmptyDatasetError(DatasetValidationError):
    """Raised when a dataset holds no examples."""


class MissingFeatureError(DatasetValidationError):
    """Raised when an example's vector is shorter than the declared features."""


class NonFiniteFeatureError(DatasetValidationError):
    """Raised when an example carries ``NaN`` or an infinity."""

    def __init__(self, row_index: int, position: int, name: str, value: float) -> None:
        """Record where the non-finite value was found.

        Args:
            row_index: Position of the offending example.
            position: Index of the offending value within the vector.
            name: Canonical name of the feature at that position.
            value: The value that was not finite.
        """
        super().__init__(
            f"Example {row_index} carries {value!r} for {name!r} at position "
            f"{position}. Non-finite values survive most arithmetic, so they "
            "must be refused rather than trained on."
        )
        self.row_index = row_index
        self.position = position
        self.name = name
        self.value = value


class UnknownLabelError(DatasetValidationError):
    """Raised when an example's label is not a trainable class."""

    def __init__(self, row_index: int, label: object) -> None:
        """Record the row and the label that was refused.

        Args:
            row_index: Position of the offending example.
            label: The label that named no trainable class.
        """
        super().__init__(
            f"Example {row_index} is labelled {label!r}, which is not one of the "
            f"trainable classes: "
            f"{', '.join(a.value for a in TRAINABLE_ATTACK_TYPES)}."
        )
        self.row_index = row_index
        self.label = label


class InsufficientClassCoverageError(DatasetValidationError):
    """Raised when a dataset cannot distinguish between classes."""

    def __init__(self, distinct: int) -> None:
        """Record how many distinct classes were found.

        Args:
            distinct: Number of distinct classes in the dataset.
        """
        super().__init__(
            f"The dataset holds {distinct} distinct class(es). A classifier needs "
            "at least two to have anything to distinguish; one class trains a "
            "model that always answers that class."
        )
        self.distinct = distinct


@dataclass(frozen=True, slots=True)
class LabeledExample:
    """One training row: a feature vector and its ground-truth class.

    Attributes:
        features: One float per entry of
            :func:`src.feature_engineering.feature_names`, in that order.
        label: The class this vector is ground truth for.
    """

    features: tuple[float, ...]
    label: AttackType


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    """A validated set of training rows sharing one feature ordering.

    Constructed only through :func:`validate_dataset`, so holding one is proof
    the rows were checked. The counts are properties of the validated rows
    rather than separate inputs, which removes the possibility of a summary that
    disagrees with the data it summarises.

    Attributes:
        examples: The rows, in dataset order.
        feature_order: Canonical feature names, positionally describing
            :attr:`LabeledExample.features`.
    """

    examples: tuple[LabeledExample, ...]
    feature_order: tuple[str, ...]

    def __len__(self) -> int:
        """Return the number of rows.

        Returns:
            int: Size of :attr:`examples`.
        """
        return len(self.examples)

    @property
    def labels(self) -> tuple[AttackType, ...]:
        """Return the distinct classes present, in first-seen order.

        Returns:
            tuple[AttackType, ...]: One entry per distinct class.
        """
        seen: dict[AttackType, None] = {}
        for example in self.examples:
            seen.setdefault(example.label, None)
        return tuple(seen)

    @property
    def class_counts(self) -> Mapping[AttackType, int]:
        """Return how many rows carry each class.

        A class represented by a handful of rows against thousands of another is
        the imbalanced-dataset case, and the count is what a reviewer needs to
        see it before training rather than after.

        Returns:
            Mapping[AttackType, int]: Rows per class, in first-seen order.
        """
        counts: dict[AttackType, int] = {}
        for example in self.examples:
            counts[example.label] = counts.get(example.label, 0) + 1
        return counts

    @property
    def missing_classes(self) -> tuple[AttackType, ...]:
        """Return trainable classes this dataset has no rows for.

        Returns:
            tuple[AttackType, ...]: Classes in declaration order that are absent.
        """
        present = set(self.labels)
        return tuple(
            attack_type
            for attack_type in TRAINABLE_ATTACK_TYPES
            if attack_type not in present
        )


def validate_examples(
    examples: Sequence[LabeledExample],
    feature_order: Sequence[str],
) -> TrainingDataset:
    """Check a set of rows against the declared feature order and class set.

    Args:
        examples: Candidate rows, in dataset order.
        feature_order: Canonical feature names describing each vector's positions.

    Returns:
        TrainingDataset: The validated rows, unchanged and in order.

    Raises:
        EmptyDatasetError: If ``examples`` is empty.
        MissingFeatureError: If a vector's length differs from the declared
            feature count.
        NonFiniteFeatureError: If a vector carries ``NaN`` or an infinity.
        UnknownLabelError: If a row's label is not a trainable class.
        InsufficientClassCoverageError: If fewer than two distinct classes are
            present.
    """
    if not examples:
        raise EmptyDatasetError("The dataset holds no examples.")

    order = tuple(feature_order)
    width = len(order)

    for index, example in enumerate(examples):
        if len(example.features) != width:
            raise MissingFeatureError(
                f"Example {index} carries {len(example.features)} value(s) but "
                f"the dataset declares {width} feature(s): {', '.join(order)}."
            )

        if example.label not in TRAINABLE_ATTACK_TYPES:
            raise UnknownLabelError(index, example.label)

        for position, value in enumerate(example.features):
            if not math.isfinite(value):
                raise NonFiniteFeatureError(
                    index, position, order[position], value
                )

    distinct = {example.label for example in examples}
    if len(distinct) < 2:
        raise InsufficientClassCoverageError(len(distinct))

    return TrainingDataset(examples=tuple(examples), feature_order=order)


def dataset_from_records(
    rows: Sequence[tuple[Sequence[float], AttackType]],
    feature_order: Sequence[str] | None = None,
) -> TrainingDataset:
    """Build a validated dataset from raw vectors and labels.

    Args:
        rows: ``(vector, label)`` pairs in dataset order.
        feature_order: Feature names describing the vectors, or ``None`` to use
            :func:`src.feature_engineering.feature_names`.

    Returns:
        TrainingDataset: The validated dataset.
    """
    order = tuple(feature_order) if feature_order is not None else feature_names()
    examples = [
        LabeledExample(features=tuple(float(value) for value in vector), label=label)
        for vector, label in rows
    ]
    return validate_examples(examples, order)


def describe_dataset(dataset: TrainingDataset) -> str:
    """Summarise a dataset for a training log line.

    Reports the class counts rather than only the row total, because a dataset of
    ten thousand rows is not usefully described as "ten thousand rows" when one
    class holds ninety-nine percent of them.

    Args:
        dataset: The validated dataset to describe.

    Returns:
        str: A one-line summary naming rows, features and class distribution.
    """
    counts = ", ".join(
        f"{attack_type.value}={count}" for attack_type, count in dataset.class_counts.items()
    )
    return (
        f"{len(dataset)} row(s), {len(dataset.feature_order)} feature(s) "
        f"[{', '.join(dataset.feature_order)}]; classes: {counts}"
    )


def describe_schema(feature_order: Sequence[str]) -> str:
    """Return the feature order as a single stable string for artifact metadata.

    Args:
        feature_order: Canonical feature names.

    Returns:
        str: The names joined by commas, in order. Empty when no order was given,
        which is a state the dataset validator rejects rather than stores.
    """
    return ",".join(feature_order)