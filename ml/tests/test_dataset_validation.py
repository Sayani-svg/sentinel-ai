"""The dataset validator is the only thing standing between a file and a model.

Its refusals are the deliverable of this phase, since there is no real dataset to
train on yet. These tests therefore check what the validator *rejects* at least as
carefully as what it accepts: every failure mode below produces a model that
either trains on nonsense or reports a confidence it has not earned.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from app.schemas.prediction import AttackType

from src.dataset_validation import (
    DatasetStorage,
    DatasetValidationError,
    EmptyDatasetError,
    InsufficientClassCoverageError,
    LabeledExample,
    MissingFeatureError,
    NonFiniteFeatureError,
    TrainingDataset,
    UnknownLabelError,
    dataset_from_records,
    describe_dataset,
    describe_schema,
    validate_arrays,
    validate_examples,
)
from src.feature_engineering import feature_names
from src.preprocessing import TRAINABLE_ATTACK_TYPES


def _vector(value: float = 1.0) -> tuple[float, ...]:
    """Return a valid vector of the declared width.

    Args:
        value: The value to fill every position with.

    Returns:
        tuple[float, ...]: One value per declared feature.
    """
    return tuple(value for _ in feature_names())


def _dataset() -> "object":
    """Return a minimal valid dataset of two classes.

    Returns:
        TrainingDataset: Two examples, one per class.
    """
    return dataset_from_records(
        [
            (_vector(1.0), AttackType.BENIGN),
            (_vector(2.0), AttackType.DOS),
        ]
    )


def test_valid_dataset_is_accepted() -> None:
    """A complete, finite, two-class dataset must validate."""
    dataset = _dataset()
    assert len(dataset) == 2
    assert dataset.feature_order == feature_names()


def test_rows_are_preserved_in_order() -> None:
    """Row order must survive validation, for reproducible splits."""
    dataset = dataset_from_records(
        [
            (_vector(1.0), AttackType.BENIGN),
            (_vector(2.0), AttackType.DOS),
            (_vector(3.0), AttackType.BENIGN),
        ]
    )
    assert [row.features[0] for row in dataset.examples] == [1.0, 2.0, 3.0]


def test_empty_dataset_is_refused() -> None:
    """An empty dataset must be refused, not silently trained on."""
    with pytest.raises(EmptyDatasetError):
        validate_examples([], feature_names())


def test_single_class_dataset_is_refused() -> None:
    """One class trains a model that always answers that class."""
    rows = [(_vector(), AttackType.BENIGN) for _ in range(10)]
    with pytest.raises(InsufficientClassCoverageError) as excinfo:
        dataset_from_records(rows)
    assert excinfo.value.distinct == 1


def test_short_vector_is_refused() -> None:
    """A vector shorter than the declared order must be refused."""
    example = LabeledExample(
        features=_vector()[:-1],
        label=AttackType.BENIGN,
    )
    with pytest.raises(MissingFeatureError):
        validate_examples([example, LabeledExample(_vector(), AttackType.DOS)], feature_names())


def test_long_vector_is_refused() -> None:
    """A vector longer than the declared order must be refused too.

    Extra positions are worse than missing ones: truncating them silently drops
    measurements the estimator was trained on.
    """
    example = LabeledExample(features=(*_vector(), 1.0), label=AttackType.BENIGN)
    with pytest.raises(MissingFeatureError):
        validate_examples([example, LabeledExample(_vector(), AttackType.DOS)], feature_names())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_refused(bad: float) -> None:
    """``NaN`` and infinities must be refused.

    They survive most arithmetic without complaint, so a model trained on them
    trains "successfully" and predicts nonsensically.
    """
    features = list(_vector())
    features[2] = bad
    example = LabeledExample(features=tuple(features), label=AttackType.BENIGN)
    with pytest.raises(NonFiniteFeatureError) as excinfo:
        validate_examples([example, LabeledExample(_vector(), AttackType.DOS)], feature_names())

    error = excinfo.value
    assert error.row_index == 0
    assert error.position == 2
    assert math.isnan(error.value) and math.isnan(bad) or error.value == bad


def test_non_finite_error_names_the_feature() -> None:
    """The refusal must name the feature, so the source column can be found."""
    features = list(_vector())
    features[1] = float("nan")
    example = LabeledExample(features=tuple(features), label=AttackType.BENIGN)
    with pytest.raises(NonFiniteFeatureError) as excinfo:
        validate_examples([example, LabeledExample(_vector(), AttackType.DOS)], feature_names())
    assert excinfo.value.name == feature_names()[1]


def test_unknown_label_class_is_refused() -> None:
    """A label outside the trainable vocabulary must be refused."""
    example = LabeledExample(features=_vector(), label=AttackType.UNKNOWN)
    with pytest.raises(UnknownLabelError) as excinfo:
        validate_examples([example, LabeledExample(_vector(), AttackType.DOS)], feature_names())
    assert excinfo.value.row_index == 0


def test_unlabelled_row_is_refused() -> None:
    """A row with no label states no ground truth, so it cannot be learned from."""
    example = LabeledExample(features=_vector(), label=None)  # type: ignore[arg-type]
    with pytest.raises(UnknownLabelError):
        validate_examples([example, LabeledExample(_vector(), AttackType.DOS)], feature_names())


def test_every_refusal_shares_one_base_class() -> None:
    """Callers should be able to catch the family, not enumerate cases."""
    for rows in (
        [],
        [LabeledExample(_vector(), AttackType.BENIGN)],
    ):
        with pytest.raises(DatasetValidationError):
            validate_examples(rows, feature_names())


def test_labels_are_reported_in_first_seen_order() -> None:
    """Distinct labels must be reported in dataset order, not sorted.

    Reporting in dataset order means the summary matches the rows as a reviewer
    reads them.
    """
    dataset = dataset_from_records(
        [
            (_vector(), AttackType.DDOS),
            (_vector(), AttackType.BENIGN),
            (_vector(), AttackType.DDOS),
        ]
    )
    assert dataset.labels == (AttackType.DDOS, AttackType.BENIGN)


def test_class_counts_are_exact() -> None:
    """Counts must reflect the rows, so imbalance is visible before training."""
    dataset = dataset_from_records(
        [
            (_vector(), AttackType.BENIGN),
            (_vector(), AttackType.BENIGN),
            (_vector(), AttackType.DOS),
        ]
    )
    assert dataset.class_counts == {AttackType.BENIGN: 2, AttackType.DOS: 1}
    assert len(dataset.class_counts) == 2


def test_missing_classes_are_reported_in_vocabulary_order() -> None:
    """Absent classes must be listed in declaration order, for stable reports."""
    dataset = _dataset()
    assert AttackType.BOT in dataset.missing_classes
    assert dataset.missing_classes == tuple(
        attack_type
        for attack_type in TRAINABLE_ATTACK_TYPES
        if attack_type not in {AttackType.BENIGN, AttackType.DOS}
    )


def test_full_coverage_reports_nothing_missing() -> None:
    """A dataset covering every class must report no gaps."""
    dataset = dataset_from_records(
        [(_vector(), attack_type) for attack_type in TRAINABLE_ATTACK_TYPES]
    )
    assert dataset.missing_classes == ()


def test_describe_reports_counts_not_just_a_total() -> None:
    """A summary must expose imbalance, not hide it behind a row count."""
    summary = describe_dataset(_dataset())
    assert "Benign" in summary and "DoS" in summary
    assert str(len(_dataset())) in summary


def test_describe_schema_preserves_order() -> None:
    """The artifact's schema string must keep the positional order intact."""
    joined = describe_schema(feature_names())
    assert joined == ",".join(feature_names())
    assert joined.split(",")[0] == feature_names()[0]


def test_describe_schema_of_nothing_is_empty() -> None:
    """An empty order must render empty, not fall back to a hidden default.

    The validator rejects an empty order, so silently substituting one here
    would hand a caller a schema string for a dataset that cannot exist.
    """
    assert describe_schema([]) == ""


def test_examples_are_immutable() -> None:
    """A validated row must not be editable after the fact.

    Mutating a feature after validation would reintroduce exactly the NaN and
    width problems the validator exists to catch.
    """
    example = LabeledExample(features=_vector(), label=AttackType.BENIGN)
    with pytest.raises(AttributeError):
        example.label = AttackType.DOS  # type: ignore[misc]


def test_validated_dataset_stores_compact_arrays() -> None:
    """Rows must be stored once as numeric arrays, not as per-example objects."""
    dataset = _dataset()

    assert dataset.feature_matrix.dtype == np.float32
    assert dataset.feature_matrix.shape == (2, len(feature_names()))
    assert dataset.label_codes.dtype == np.uint8
    assert dataset.storage is DatasetStorage.MEMORY


def test_arrays_can_be_validated_directly() -> None:
    """A streaming reader must be able to build a dataset without Python rows."""
    features = np.full((2, len(feature_names())), 3.0, dtype=np.float32)
    codes = np.array([0, 1], dtype=np.uint8)

    dataset = validate_arrays(features, codes, feature_names())

    assert len(dataset) == 2
    assert dataset.labels == (AttackType.BENIGN, AttackType.DOS)
    assert dataset.examples[0].features == _vector(3.0)


def test_disk_backed_dataset_preserves_the_row_contract(tmp_path: Path) -> None:
    """Memory-mapped arrays must look identical to resident arrays for callers."""
    dataset = _dataset().to_disk(tmp_path, name="checkpoint")

    assert dataset.storage is DatasetStorage.DISK
    assert len(dataset.storage_paths) == 2
    assert all(path.exists() for path in dataset.storage_paths)
    assert dataset.examples[0].features == _vector(1.0)
    assert dataset.examples[1].features == _vector(2.0)


def test_concat_combines_arrays_without_materialising_examples(tmp_path: Path) -> None:
    """Several validated batches must form one compact dataset."""
    first = _dataset()
    second = dataset_from_records(
        [(_vector(1.0), AttackType.BENIGN), (_vector(2.0), AttackType.DOS)]
    )

    combined = TrainingDataset.concat(
        [first, second], storage=DatasetStorage.DISK, storage_dir=tmp_path
    )

    assert len(combined) == 4
    assert combined.feature_matrix.shape == (4, len(feature_names()))
    assert combined.class_counts == {AttackType.BENIGN: 2, AttackType.DOS: 2}
    assert combined.storage is DatasetStorage.DISK


def test_noncanonical_feature_order_is_refused() -> None:
    """Five columns are not enough: the names and their order are the contract."""
    features = np.full((2, len(feature_names())), 3.0, dtype=np.float32)
    codes = np.array([0, 1], dtype=np.uint8)

    with pytest.raises(MissingFeatureError, match="canonical"):
        validate_arrays(features, codes, tuple(reversed(feature_names())))


def test_validated_arrays_are_read_only_copies() -> None:
    """Validation must not hand callers a mutable alias to its invariants."""
    features = np.full((2, len(feature_names())), 3.0, dtype=np.float32)
    codes = np.array([0, 1], dtype=np.uint8)

    dataset = validate_arrays(features, codes, feature_names())
    features[0, 0] = 99.0
    codes[0] = 7

    assert dataset.feature_matrix[0, 0] == 3.0
    assert dataset.label_codes[0] == 0
    assert not dataset.feature_matrix.flags.writeable
    assert not dataset.label_codes.flags.writeable
    with pytest.raises(ValueError):
        dataset.feature_matrix[0, 0] = 1.0


def test_lazy_examples_preserve_tuple_behavior() -> None:
    """The compact view must still compare and hash like the old tuple."""
    dataset = _dataset()
    expected = tuple(dataset.examples)

    assert dataset.examples == expected
    assert hash(dataset.examples) == hash(expected)
    assert repr(dataset.examples) == repr(expected)
    assert dataset.examples[1:] == expected[1:]


def test_cleanup_removes_every_dataset_file(tmp_path: Path) -> None:
    """A disk-backed dataset must know how to remove its own files."""
    dataset = _dataset().to_disk(tmp_path, name="checkpoint")
    paths = dataset.storage_paths
    assert all(path.exists() for path in paths)

    dataset.cleanup()

    assert not any(path.exists() for path in paths)
    assert not list(tmp_path.glob(".*checkpoint*"))
