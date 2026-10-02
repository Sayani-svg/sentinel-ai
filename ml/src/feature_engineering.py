"""Feature vectors for training and serving, drawn from the existing parsed log.

Nothing here decides what a "feature" is. The vocabulary is
:class:`app.services.prediction_service.DetectionFeature`, the same five numeric
measurements the rule-based detector reads, and the records arrive as
:class:`~app.services.prediction_service.DetectionRecord`, the same shape the
detector is handed. A training pipeline that invented its own feature set would
train on columns the detector never looks at, and the resulting artifact would be
unusable at serving time without a translation layer nobody would remember to
maintain.

**Column order is part of the contract.** :func:`numeric_features` derives the
ordering from the enum's own declaration order rather than from a list written
out here, so a feature added to the enum is picked up automatically and no copy
can fall out of step. The order still has to be *recorded* alongside a trained
artifact, because a vector is positional: serving a model against a differently
ordered vector produces confident nonsense rather than an error.
:func:`feature_names` is what gets recorded, and
:func:`assert_feature_order` is what serving checks.

**Absent is not zero.** A ``DetectionRecord`` omits a feature it could not read,
so that a missing measurement is never mistaken for a measured absence. Building
a dense vector therefore has to do something with those gaps. This module
refuses rather than imputing, because the right value is a property of the
dataset rather than of the code: filling with ``0.0`` would assert that a flow
lasted no time and moved no bytes, which is a confident statement about traffic
that was merely unmeasured. :func:`missing_features` reports which columns are
absent so an operator can see why rows were refused, and imputation belongs in a
preprocessing step added with the real dataset and recorded in the artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from app.services.prediction_service import (
    DetectionFeature,
    DetectionInput,
    DetectionRecord,
    to_detection_input,
)
from app.services.log_parser import ParsedLog

#: The numeric features, in declaration order, excluding the label column.
#: Derived rather than written out so the two cannot diverge.
NUMERIC_FEATURES: Final[tuple[DetectionFeature, ...]] = tuple(
    feature for feature in DetectionFeature if feature is not DetectionFeature.LABEL
)


class FeatureExtractionError(ValueError):
    """Base class for failures turning records into feature vectors."""


class MissingFeatureError(FeatureExtractionError):
    """Raised when a record lacks a feature the vector must carry."""

    def __init__(self, row_index: int, missing: Sequence[str]) -> None:
        """Record which row lacked which features.

        Args:
            row_index: Position of the offending record in the dataset.
            missing: Canonical feature names the record did not carry.
        """
        super().__init__(
            f"Record {row_index} is missing {', '.join(missing)}. "
            "A feature vector must carry every declared feature; this module "
            "does not impute, because the correct fill is a property of the "
            "dataset rather than of the code."
        )
        self.row_index = row_index
        self.missing = tuple(missing)


class FeatureOrderError(FeatureExtractionError):
    """Raised when a feature order does not match the one a model was built with."""


def feature_names() -> tuple[str, ...]:
    """Return the canonical feature names in vector order.

    This tuple is what a trained artifact records. Positions in a feature vector
    are meaningless without it.

    Returns:
        tuple[str, ...]: One name per position, in declaration order.
    """
    return tuple(str(feature) for feature in NUMERIC_FEATURES)


def missing_features(record: DetectionRecord) -> tuple[str, ...]:
    """Return the declared features a record does not carry.

    Args:
        record: The prepared record to inspect.

    Returns:
        tuple[str, ...]: Names absent from the record, in vector order. Empty
        when the record is complete.
    """
    return tuple(name for name in feature_names() if name not in record.features)


def to_feature_vector(record: DetectionRecord, *, row_index: int = 0) -> tuple[float, ...]:
    """Return one record as a dense feature vector.

    Args:
        record: A prepared record from
            :func:`app.services.prediction_service.build_detection_input`.
        row_index: Position of the record in the dataset, used only to make a
            refusal identifiable.

    Returns:
        tuple[float, ...]: One float per entry of :func:`feature_names`, in that
        order.

    Raises:
        MissingFeatureError: If the record omits any declared feature.
    """
    absent = missing_features(record)
    if absent:
        raise MissingFeatureError(row_index, absent)
    return tuple(float(record.features[name]) for name in feature_names())


def to_feature_matrix(
    records: Sequence[DetectionRecord],
) -> tuple[tuple[float, ...], ...]:
    """Return records as a row-major feature matrix.

    Args:
        records: Prepared records in dataset order.

    Returns:
        tuple[tuple[float, ...], ...]: One vector per record, each positionally
        aligned with :func:`feature_names`.

    Raises:
        MissingFeatureError: If any record omits a declared feature. The row
            index in the error identifies which one.
    """
    return tuple(
        to_feature_vector(record, row_index=index)
        for index, record in enumerate(records)
    )


def to_feature_space(parsed: ParsedLog) -> DetectionInput:
    """Return a parsed log file in the feature space both training and serving use.

    Args:
        parsed: A parsed log file, as
            :mod:`app.services.log_parser` produces it.

    Returns:
        DetectionInput: Records reduced to the canonical numeric features.

    Raises:
        app.services.prediction_service.PredictionServiceError: If the file
        yielded no records.
    """
    return to_detection_input(parsed)


def assert_feature_order(expected: Sequence[str]) -> None:
    """Check that a model's feature order matches this build's.

    Called before serving so a model trained against a different column order is
    refused outright. Positional features produce confident wrong answers rather
    than errors, so this check is the only thing standing between a reordered
    artifact and a stream of plausible nonsense.

    Args:
        expected: The feature order the model was trained with.

    Raises:
        FeatureOrderError: If it differs from :func:`feature_names` in content or
            in order.
    """
    current = feature_names()
    recorded = tuple(expected)
    if recorded == current:
        return

    if sorted(recorded) == sorted(current):
        raise FeatureOrderError(
            "The model was trained with the same features in a different order: "
            f"trained {recorded}, this build {current}. Positional features would "
            "silently produce wrong predictions."
        )
    raise FeatureOrderError(
        f"The model was trained with features this build does not define: "
        f"trained {recorded}, this build {current}."
    )


def feature_coverage(records: Sequence[DetectionRecord]) -> Mapping[str, int]:
    """Report how many records carried each declared feature.

    Intended for dataset inspection. A column present in the file but missing
    from most rows is the usual reason :func:`to_feature_vector` refuses rows,
    and this is how that shows up before training is attempted.

    Args:
        records: Prepared records from a dataset file.

    Returns:
        Mapping[str, int]: Feature name to the number of records carrying it, in
        vector order.
    """
    return {
        name: sum(1 for record in records if name in record.features)
        for name in feature_names()
    }