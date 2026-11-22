"""The training dataset contract, and the rules a dataset must satisfy.

This module defines what a training dataset *is* and refuses datasets that cannot
produce an honest model. It deliberately does not create one. There is no data
in ``ml/dataset`` -- the directory holds a ``.gitkeep`` -- and fabricating rows
to demonstrate a training run would produce a model whose reported accuracy
describes the fabrication. The checks below are therefore the deliverable: they
say what will be accepted, so the checks can be run against a real benchmark the
moment one is placed.

A dataset here is a compact pair of arrays: a ``float32`` feature matrix and a
``uint8`` label-code vector. :class:`LabeledExample` survives as a lazy view for
callers that read rows one at a time, but it is no longer the storage: a few
million Python objects would outweigh the training matrix several times over.
The feature ordering is stated once for the whole dataset, because a vector is
positional and a per-row ordering could disagree with itself unnoticed.

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
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from app.schemas.prediction import AttackType

from src.feature_engineering import feature_names
from src.preprocessing import (
    CLASS_COUNT,
    INDEX_TO_LABEL,
    TRAINABLE_ATTACK_TYPES,
    index_to_label,
    label_to_index,
)


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


class IncompleteDatasetError(DatasetValidationError):
    """Raised when a dataset is a partial view of its source rows."""


@dataclass(frozen=True, slots=True)
class DatasetCompleteness:
    """Whether a dataset represents a complete source load.

    Attributes:
        complete: ``False`` when a source was capped, skipped or otherwise only
            partially retained.
        truncated_paths: Source files whose rows were not fully retained.
        reasons: Human-readable explanations of the incomplete load.
    """

    complete: bool = True
    truncated_paths: tuple[Path, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def incomplete(self) -> bool:
        """Return whether the load was incomplete."""
        return not self.complete

    @classmethod
    def merge(cls, items: Sequence["DatasetCompleteness"]) -> "DatasetCompleteness":
        """Combine completeness records from several source files."""
        paths = tuple(dict.fromkeys(path for item in items for path in item.truncated_paths))
        reasons = tuple(dict.fromkeys(reason for item in items for reason in item.reasons))
        return cls(
            complete=all(item.complete for item in items),
            truncated_paths=paths,
            reasons=reasons,
        )


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


class DatasetStorage(StrEnum):
    """Where a validated dataset's arrays live."""

    MEMORY = "memory"
    DISK = "disk"

    @classmethod
    def parse(cls, value: "DatasetStorage | str") -> "DatasetStorage":
        """Accept either an enum member or its string spelling."""
        return value if isinstance(value, cls) else cls(value)


def close_memmap(array: np.ndarray) -> None:
    """Close the mapping behind an array or any view derived from it."""
    candidate: object | None = array
    while candidate is not None:
        handle = getattr(candidate, "_mmap", None)
        if handle is not None:
            handle.close()
            return
        candidate = getattr(candidate, "base", None)


def _write_npy_atomically(path: Path, array: np.ndarray) -> None:
    """Write one ``.npy`` file through a temporary file in the same directory."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _create_memmap_atomically(
    path: Path, *, dtype: np.dtype, shape: tuple[int, ...]
) -> tuple[np.memmap, Path]:
    """Create a temporary ``.npy`` memmap whose final name is published later."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        array = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=dtype, shape=shape
        )
        return array, temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class _LazyExamples(Sequence[LabeledExample]):
    """A read-only sequence view over a dataset's compact arrays."""

    __slots__ = ("_features", "_codes")

    def __init__(self, dataset: "TrainingDataset") -> None:
        self._features = dataset.feature_matrix
        self._codes = dataset.label_codes

    def __len__(self) -> int:
        return int(self._features.shape[0])

    def __getitem__(self, index: int | slice) -> LabeledExample | tuple[LabeledExample, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        return LabeledExample(
            features=tuple(float(value) for value in self._features[index]),
            label=index_to_label(int(self._codes[index])),
        )

    def __iter__(self) -> Iterator[LabeledExample]:
        for features, code in zip(self._features, self._codes, strict=True):
            yield LabeledExample(
                features=tuple(float(value) for value in features),
                label=index_to_label(int(code)),
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence) or isinstance(other, (str, bytes)):
            return NotImplemented
        return tuple(self) == tuple(other)

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __add__(self, other: Sequence[LabeledExample]) -> tuple[LabeledExample, ...]:
        return tuple(self) + tuple(other)

    def __radd__(self, other: Sequence[LabeledExample]) -> tuple[LabeledExample, ...]:
        return tuple(other) + tuple(self)

    def __repr__(self) -> str:
        return repr(tuple(self))


@dataclass(frozen=True, slots=True, eq=False)
class TrainingDataset:
    """A validated set of training rows sharing one feature ordering.

    Constructed only through :func:`validate_examples` or
    :func:`validate_arrays`, so holding one is proof the rows were checked. The
    counts are derived once at construction rather than recomputed from the rows
    by every caller.

    Attributes:
        feature_order: Canonical feature names, positionally describing columns.
        completeness: Whether the rows represent a complete source load.
        storage: Whether the arrays are resident or memory-mapped from disk.
        storage_paths: Files backing a memory-mapped dataset, if any.
        cleanup_paths: Every generated file owned by this dataset, including
            side files that are not array backing files.
    """

    _features: np.ndarray
    _codes: np.ndarray
    feature_order: tuple[str, ...]
    completeness: DatasetCompleteness = field(default_factory=DatasetCompleteness)
    _labels: tuple[AttackType, ...] = ()
    _counts: Mapping[AttackType, int] = field(default_factory=dict)
    storage: DatasetStorage = DatasetStorage.MEMORY
    storage_paths: tuple[Path, ...] = ()
    cleanup_paths: tuple[Path, ...] = ()

    def __len__(self) -> int:
        return int(self._features.shape[0])

    @property
    def examples(self) -> Sequence[LabeledExample]:
        """Return the rows as a lazy, read-only sequence."""
        return _LazyExamples(self)

    @property
    def feature_matrix(self) -> np.ndarray:
        """Return the validated ``float32`` matrix, in feature-order columns."""
        return self._features

    @property
    def label_codes(self) -> np.ndarray:
        """Return the validated ``uint8`` label-code vector."""
        return self._codes

    def cleanup(self) -> None:
        """Close any memory maps and remove every file this dataset owns."""
        close_memmap(self._features)
        close_memmap(self._codes)
        for path in dict.fromkeys((*self.storage_paths, *self.cleanup_paths)):
            path.unlink(missing_ok=True)

    @property
    def labels(self) -> tuple[AttackType, ...]:
        """Return the distinct classes present, in first-seen order."""
        return self._labels

    @property
    def class_counts(self) -> Mapping[AttackType, int]:
        """Return how many rows carry each class, in first-seen order."""
        return self._counts

    @property
    def missing_classes(self) -> tuple[AttackType, ...]:
        """Return trainable classes this dataset has no rows for."""
        present = set(self._labels)
        return tuple(
            attack_type
            for attack_type in TRAINABLE_ATTACK_TYPES
            if attack_type not in present
        )

    def to_disk(self, directory: Path, *, name: str = "dataset") -> "TrainingDataset":
        """Copy the arrays into memory-mapped ``.npy`` files."""
        directory.mkdir(parents=True, exist_ok=True)
        feature_path = directory / f"{name}.features.npy"
        code_path = directory / f"{name}.labels.npy"
        written: list[Path] = []
        try:
            _write_npy_atomically(feature_path, self._features)
            written.append(feature_path)
            _write_npy_atomically(code_path, self._codes)
            written.append(code_path)
            features = np.load(feature_path, mmap_mode="r")
            codes = np.load(code_path, mmap_mode="r")
            return TrainingDataset(
                _features=features,
                _codes=codes,
                feature_order=self.feature_order,
                completeness=self.completeness,
                _labels=self._labels,
                _counts=self._counts,
                storage=DatasetStorage.DISK,
                storage_paths=(feature_path, code_path),
                cleanup_paths=(feature_path, code_path),
            )
        except BaseException:
            if "features" in locals():
                close_memmap(features)
            if "codes" in locals():
                close_memmap(codes)
            for path in written:
                path.unlink(missing_ok=True)
            raise

    @classmethod
    def concat(
        cls,
        datasets: Sequence["TrainingDataset"],
        *,
        storage: DatasetStorage | str = DatasetStorage.MEMORY,
        storage_dir: Path | None = None,
        name: str = "dataset",
        completeness: DatasetCompleteness | None = None,
    ) -> "TrainingDataset":
        """Combine datasets without converting their rows back to objects."""
        items = tuple(datasets)
        if not items:
            raise EmptyDatasetError("There are no datasets to combine.")
        order = items[0].feature_order
        expected = feature_names()
        if order != expected:
            raise MissingFeatureError(
                f"Datasets must use the canonical feature order {expected}, got "
                f"{order}."
            )
        if any(item.feature_order != expected for item in items[1:]):
            raise MissingFeatureError("Datasets disagree about feature order.")

        rows = sum(len(item) for item in items)
        width = len(expected)
        labels = tuple(dict.fromkeys(label for item in items for label in item.labels))
        counts: dict[AttackType, int] = {}
        for item in items:
            for label, count in item.class_counts.items():
                counts[label] = counts.get(label, 0) + count
        merged = completeness or DatasetCompleteness.merge(
            [item.completeness for item in items]
        )
        inherited = tuple(
            dict.fromkeys(path for item in items for path in item.cleanup_paths)
        )
        mode = DatasetStorage.parse(storage)

        temporary_paths: list[Path] = []
        final_paths: list[Path] = []
        open_arrays: list[np.ndarray] = []
        feature_path: Path | None = None
        code_path: Path | None = None
        try:
            if mode is DatasetStorage.DISK:
                if storage_dir is None:
                    raise ValueError("storage_dir is required for disk-backed datasets.")
                storage_dir.mkdir(parents=True, exist_ok=True)
                feature_path = storage_dir / f"{name}.features.npy"
                code_path = storage_dir / f"{name}.labels.npy"
                writable_features, feature_temp = _create_memmap_atomically(
                    feature_path, dtype=np.float32, shape=(rows, width)
                )
                temporary_paths.append(feature_temp)
                open_arrays.append(writable_features)
                writable_codes, code_temp = _create_memmap_atomically(
                    code_path, dtype=np.uint8, shape=(rows,)
                )
                temporary_paths.append(code_temp)
                open_arrays.append(writable_codes)
                features = writable_features
                codes = writable_codes
            else:
                features = np.empty((rows, width), dtype=np.float32)
                codes = np.empty(rows, dtype=np.uint8)

            offset = 0
            for item in items:
                size = len(item)
                features[offset : offset + size] = item.feature_matrix
                codes[offset : offset + size] = item.label_codes
                offset += size

            if mode is DatasetStorage.DISK:
                features.flush()
                codes.flush()
                close_memmap(features)
                close_memmap(codes)
                open_arrays.clear()
                os.replace(feature_temp, feature_path)
                final_paths.append(feature_path)
                os.replace(code_temp, code_path)
                final_paths.append(code_path)
                features = np.load(feature_path, mmap_mode="r")
                codes = np.load(code_path, mmap_mode="r")
                open_arrays.extend((features, codes))

            features.flags.writeable = False
            codes.flags.writeable = False
            storage_paths = (
                (feature_path, code_path)
                if feature_path is not None and code_path is not None
                else ()
            )
            cleanup_paths = tuple(dict.fromkeys((*inherited, *storage_paths)))
            return cls(
                _features=features,
                _codes=codes,
                feature_order=expected,
                completeness=merged,
                _labels=labels,
                _counts=counts,
                storage=mode,
                storage_paths=storage_paths,
                cleanup_paths=cleanup_paths,
            )
        except BaseException:
            for array in open_arrays:
                close_memmap(array)
            for path in temporary_paths:
                path.unlink(missing_ok=True)
            for path in final_paths:
                path.unlink(missing_ok=True)
            raise


def validate_examples(
    examples: Sequence[LabeledExample],
    feature_order: Sequence[str],
    *,
    completeness: DatasetCompleteness | None = None,
    require_class_coverage: bool = True,
) -> TrainingDataset:
    """Check a set of rows against the declared feature order and class set.

    Args:
        examples: Candidate rows, in dataset order.
        feature_order: Canonical feature names describing each vector's positions.
        completeness: Source-load metadata carried onto the validated dataset.
        require_class_coverage: Whether this batch alone must hold at least two
            classes. A CIC capture file is only part of a logical dataset, so its
            rows are checked for features, finiteness and labels here, while class
            coverage is enforced once the selected files are combined.

    Returns:
        TrainingDataset: The validated rows, unchanged and in order.

    Raises:
        EmptyDatasetError: If ``examples`` is empty.
        MissingFeatureError: If a vector's length differs from the declared
            feature count.
        NonFiniteFeatureError: If a vector carries ``NaN`` or an infinity.
        UnknownLabelError: If a row's label is not a trainable class.
        InsufficientClassCoverageError: If ``require_class_coverage`` is set and
            fewer than two distinct classes are present.
    """
    if not examples:
        raise EmptyDatasetError("The dataset holds no examples.")

    order = tuple(feature_order)
    width = len(order)
    features = np.empty((len(examples), width), dtype=np.float32)
    codes = np.empty(len(examples), dtype=np.uint8)

    for index, example in enumerate(examples):
        if len(example.features) != width:
            raise MissingFeatureError(
                f"Example {index} carries {len(example.features)} value(s) but "
                f"the dataset declares {width} feature(s): {', '.join(order)}."
            )
        if example.label not in TRAINABLE_ATTACK_TYPES:
            raise UnknownLabelError(index, example.label)
        code = label_to_index(example.label)
        for position, value in enumerate(example.features):
            if not math.isfinite(value):
                raise NonFiniteFeatureError(index, position, order[position], value)
            features[index, position] = value
        codes[index] = code

    overflow = ~np.isfinite(features)
    if overflow.any():
        row, position = (int(part) for part in np.argwhere(overflow)[0])
        raise NonFiniteFeatureError(
            row, position, order[position], float(features[row, position])
        )

    return validate_arrays(
        features,
        codes,
        order,
        completeness=completeness,
        require_class_coverage=require_class_coverage,
    )


def validate_arrays(
    features: np.ndarray,
    label_codes: np.ndarray,
    feature_order: Sequence[str],
    *,
    completeness: DatasetCompleteness | None = None,
    require_class_coverage: bool = True,
    copy_arrays: bool = True,
) -> TrainingDataset:
    """Validate compact feature and label-code arrays.

    Args:
        features: ``(rows, features)`` matrix of float values.
        label_codes: ``(rows,)`` vector of integer class codes.
        feature_order: Canonical feature names describing matrix columns.
        completeness: Source-load metadata carried onto the dataset.
        require_class_coverage: Whether at least two classes must be present.
        copy_arrays: Whether to copy caller-owned arrays before retaining them.
            Internal streaming readers pass ``False`` for arrays they created.

    Returns:
        TrainingDataset: A validated, compact dataset.

    Raises:
        EmptyDatasetError: If the arrays hold no rows.
        MissingFeatureError: If their shapes disagree with the feature order.
        NonFiniteFeatureError: If a matrix value is not finite.
        UnknownLabelError: If a label code is outside the trainable vocabulary.
        InsufficientClassCoverageError: If fewer than two classes are present.
    """
    order = tuple(feature_order)
    expected = feature_names()
    if order != expected:
        raise MissingFeatureError(
            f"Feature order must be exactly the canonical order {expected}, got {order}."
        )
    width = len(expected)
    matrix = np.asarray(features)
    codes = np.asarray(label_codes)

    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise MissingFeatureError(
            f"The feature matrix has shape {matrix.shape}, but the dataset "
            f"declares {width} feature(s): {', '.join(order)}."
        )
    rows = int(matrix.shape[0])
    if rows == 0:
        raise EmptyDatasetError("The dataset holds no examples.")
    if codes.ndim != 1 or int(codes.shape[0]) != rows:
        raise MissingFeatureError(
            f"The label vector has shape {codes.shape}, but the feature matrix "
            f"has {rows} row(s)."
        )

    if not np.issubdtype(codes.dtype, np.integer):
        raise UnknownLabelError(0, codes.flat[0])
    invalid = (codes < 0) | (codes >= CLASS_COUNT)
    if invalid.any():
        index = int(np.argmax(invalid))
        raise UnknownLabelError(index, codes[index])
    try:
        if copy_arrays:
            codes8 = np.array(codes, dtype=np.uint8, copy=True)
            matrix32 = np.array(matrix, dtype=np.float32, copy=True)
        else:
            codes8 = np.asarray(codes, dtype=np.uint8)
            matrix32 = np.asarray(matrix, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise MissingFeatureError("Feature values must be numeric.") from exc
    matrix32.flags.writeable = False
    codes8.flags.writeable = False
    non_finite = ~np.isfinite(matrix32)
    if non_finite.any():
        row, position = (int(part) for part in np.argwhere(non_finite)[0])
        raise NonFiniteFeatureError(
            row, position, order[position], float(matrix32[row, position])
        )

    present_codes, first_positions = np.unique(codes8, return_index=True)
    present_codes = tuple(
        int(code) for code in present_codes[np.argsort(first_positions)]
    )
    if require_class_coverage and len(present_codes) < 2:
        raise InsufficientClassCoverageError(len(present_codes))

    frequency = np.bincount(codes8, minlength=CLASS_COUNT)
    labels = tuple(INDEX_TO_LABEL[code] for code in present_codes)
    counts = {
        INDEX_TO_LABEL[int(code)]: int(count)
        for code, count in enumerate(frequency)
        if count
    }
    return TrainingDataset(
        _features=matrix32,
        _codes=codes8,
        feature_order=order,
        completeness=completeness or DatasetCompleteness(),
        _labels=labels,
        _counts=counts,
    )


def dataset_from_records(
    rows: Sequence[tuple[Sequence[float], AttackType]],
    feature_order: Sequence[str] | None = None,
    *,
    completeness: DatasetCompleteness | None = None,
) -> TrainingDataset:
    """Build a validated dataset from raw vectors and labels.

    Args:
        rows: ``(vector, label)`` pairs in dataset order.
        feature_order: Feature names describing the vectors, or ``None`` to use
            :func:`src.feature_engineering.feature_names`.
        completeness: Source-load metadata carried onto the validated dataset.

    Returns:
        TrainingDataset: The validated dataset.
    """
    order = tuple(feature_order) if feature_order is not None else feature_names()
    examples = [
        LabeledExample(features=tuple(float(value) for value in vector), label=label)
        for vector, label in rows
    ]
    return validate_examples(examples, order, completeness=completeness)


def require_complete_dataset(
    dataset: TrainingDataset, *, allow_incomplete: bool = False
) -> None:
    """Refuse a partial dataset unless the caller explicitly allows it.

    Args:
        dataset: Dataset about to be trained or evaluated.
        allow_incomplete: Explicit opt-in for incomplete source loads.

    Raises:
        IncompleteDatasetError: If the dataset is partial without opt-in.
    """
    if dataset.completeness.incomplete and not allow_incomplete:
        paths = ", ".join(str(path) for path in dataset.completeness.truncated_paths)
        detail = f" Truncated sources: {paths}." if paths else ""
        reasons = "; ".join(dataset.completeness.reasons)
        reason_detail = f" Reasons: {reasons}." if reasons else ""
        raise IncompleteDatasetError(
            "The dataset is an incomplete view of its source rows and cannot be "
            f"used by default.{detail}{reason_detail} Set "
            "allow_incomplete_dataset=True to allow it explicitly."
        )


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