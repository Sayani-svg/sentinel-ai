"""Reading a real benchmark file into the training feature space.

The loader does not have its own CSV or JSON reader. It hands the file to
:mod:`app.services.log_parser`, the same parser the ingestion endpoint uses, so a
dataset file is read under exactly the rules a live upload would be: the same
delimiter sniffing, the same type inference, the same cap on retained rows. A
separate reader here would be a second definition of what a well-formed log file
is, and the two would disagree about exactly the malformed files that matter.

The target format is CICIDS2017 CSVs, which carry a ``Label`` column and the flow
measurements the detector recognises. That is a statement about the label
vocabulary the project already committed to in
:attr:`app.schemas.prediction.AttackType`; it is not a schema invented here. The
loader accepts whatever columns the file has and reports which ones were
understood, so a benchmark export with unfamiliar headers produces a clear
"no usable features" error rather than a silently empty dataset.

**Nothing is generated.** There is no sample dataset, no synthetic fallback, and
no row fabricated to demonstrate the pipeline. A benchmark file must be placed in
``ml/dataset`` by an operator; until then this module raises
:class:`EmptyDatasetError` and says where it looked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.core.logger import get_logger
from app.services.log_parser import ParsedLog, parse_log_file
from app.services.prediction_service import DetectionInput
from app.services.upload_service import resolve_log_format

from src.config import PATHS, MLPaths
from src.feature_engineering import (
    feature_coverage,
    feature_names,
    to_feature_space,
    to_feature_vector,
)
from src.dataset_validation import LabeledExample, TrainingDataset, validate_examples
from src.preprocessing import require_trainable_label

logger = get_logger(__name__)

#: Extensions the loader will hand to the parser. Exactly the extensions with an
#: implemented reader in :class:`app.schemas.upload.LogFileFormat` -- ``.csv`` and
#: ``.json``. ``.jsonl`` is deliberately absent: the parser has a line-delimited
#: routine internally, but no format value selects it, so offering it here would
#: advertise a loader the application cannot actually use.
SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".csv", ".json"})


class DatasetLoadError(RuntimeError):
    """Raised when a dataset file cannot be turned into training rows."""


class NoUsableFeaturesError(DatasetLoadError):
    """Raised when none of the declared features were readable in a file."""

    def __init__(self, path: Path, mapped: dict[str, str], columns: Sequence[str]) -> None:
        """Record what the file did and did not contain.

        Args:
            path: The file that was read.
            mapped: Canonical features that were resolved to a source column.
            columns: Every column the file actually had.
        """
        super().__init__(
            f"{path.name} declares none of the features this build reads. "
            f"Expected headers matching any of: {', '.join(feature_names())}. "
            f"The file has: {', '.join(columns) or 'no columns'}."
        )
        self.path = path
        self.mapped = mapped
        self.columns = tuple(columns)


class UnlabelledDatasetError(DatasetLoadError):
    """Raised when a training file carries no usable class label."""

    def __init__(self, path: Path, labels: Sequence[str]) -> None:
        """Record the file and the labels it did carry.

        Args:
            path: The file that was read.
            labels: Distinct labels found, which were none that could be used.
        """
        super().__init__(
            f"{path.name} carries no usable class label. Training data must state "
            "its own ground truth: a file whose rows are unlabelled can be "
            "classified but never learned from."
        )
        self.path = path
        self.labels = tuple(labels)


@dataclass(frozen=True, slots=True)
class LoadedFile:
    """One benchmark file reduced to training rows.

    Attributes:
        path: The file that was read.
        parsed: The parsed log, kept so column summaries remain available.
        mapped_columns: Canonical feature name to the source column it came from.
        dataset: The validated training rows.
        labels: Distinct labels the file stated, after resolution to classes.
    """

    path: Path
    parsed: ParsedLog
    mapped_columns: dict[str, str]
    dataset: TrainingDataset
    labels: tuple[str, ...]


def discover_dataset_files(
    paths: MLPaths | None = None, *, suffixes: frozenset[str] | None = None
) -> tuple[Path, ...]:
    """Return the dataset files present in the dataset directory.

    A benchmark arrives as many files, one per capture day, and they are read
    together. Order is sorted so a run over an unchanged directory is
    reproducible.

    Args:
        paths: Directories to search, or ``None`` for the configured ones.
        suffixes: Extensions to accept, or ``None`` for
            :data:`SUPPORTED_SUFFIXES`.

    Returns:
        tuple[Path, ...]: Matching files, sorted by name.
    """
    resolved = PATHS if paths is None else paths
    accepted = SUPPORTED_SUFFIXES if suffixes is None else suffixes
    if not resolved.dataset_dir.is_dir():
        return ()
    return tuple(
        sorted(
            entry
            for entry in resolved.dataset_dir.iterdir()
            if entry.is_file()
            and entry.suffix.lower() in accepted
            and not entry.name.startswith(".")
        )
    )


def parse_dataset_file(path: Path) -> ParsedLog:
    """Read one benchmark file through the application's own log parser.

    Args:
        path: The file to read.

    Returns:
        ParsedLog: The parsed records and column summaries.

    Raises:
        DatasetLoadError: If the file's extension is not one the parser handles.
        app.services.log_parser.LogParsingError: If the contents do not follow
            the layout the extension promises.
    """
    try:
        log_format = resolve_log_format(path.name, _reader_settings())
    except Exception as exc:  # noqa: BLE001 - re-raised as a dataset-level error
        raise DatasetLoadError(
            f"{path.name} is not a format this build can read: {exc}"
        ) from exc
    return parse_log_file(path, log_format)


def _reader_settings():
    """Return settings carrying only the extension allowlist the loader needs.

    The parser's format resolution reads ``SUPPORTED_UPLOAD_EXTENSIONS``. Building
    the whole application ``Settings`` would demand ``DATABASE_URL`` and
    ``SECRET_KEY`` for an offline job, so this narrow stand-in carries the one
    setting that is consulted and nothing else.

    Returns:
        Settings: A settings object with just the extension allowlist populated.
    """
    from app.core.config import Settings

    return Settings.model_construct(
        SUPPORTED_UPLOAD_EXTENSIONS=tuple(sorted(SUPPORTED_SUFFIXES))
    )


def build_examples(
    detection_input: DetectionInput,
    *,
    path: Path,
    columns: Sequence[str],
    require_labels: bool = True,
) -> tuple[LabeledExample, ...]:
    """Turn prepared records into training rows.

    Args:
        detection_input: A :class:`~app.services.prediction_service.DetectionInput`
            from :func:`src.feature_engineering.to_feature_space`.
        path: The file the records came from, used only in refusal messages.
        columns: Every column the file had, used only in refusal messages.
        require_labels: Whether rows without a usable label are refused. True for
            training data; False would be for scoring an unlabelled file, which
            this module does not do.

    Returns:
        tuple[LabeledExample, ...]: One example per record.

    Raises:
        UnlabelledDatasetError: If a record has no resolvable label.
        NoUsableFeaturesError: If no declared feature was readable.
        src.preprocessing.LabelMappingError: If a label names no trainable class.
        src.dataset_validation.DatasetValidationError: If the rows fail
            validation for any other reason.
        src.feature_engineering.MissingFeatureError: If a record omits a declared
            feature.
    """
    coverage = feature_coverage(detection_input.records)
    if not any(coverage.values()):
        raise NoUsableFeaturesError(
            path, dict(detection_input.mapped_columns), columns
        )

    examples: list[LabeledExample] = []
    for index, record in enumerate(detection_input.records):
        if record.label is None:
            if require_labels:
                raise UnlabelledDatasetError(path, detection_input.labels)
            continue
        examples.append(
            LabeledExample(
                features=to_feature_vector(record, row_index=index),
                label=require_trainable_label(record.label),
            )
        )

    if not examples and require_labels:
        # Only a problem when labels were required. A caller that explicitly
        # allowed unlabelled rows to be skipped gets an empty result and decides
        # for itself whether that is acceptable.
        raise UnlabelledDatasetError(path, detection_input.labels)

    return tuple(examples)


def load_dataset_file(
    path: Path, *, require_labels: bool = True
) -> LoadedFile:
    """Read one benchmark file and return it as a validated training dataset.

    Args:
        path: The file to read.
        require_labels: Whether rows without a usable label are refused.

    Returns:
        LoadedFile: The parsed file, its column mapping and its validated rows.

    Raises:
        DatasetLoadError: If the file cannot be read, declares none of the
            features this build reads, or carries no usable labels.
        src.dataset_validation.DatasetValidationError: If the rows fail
            validation.
    """
    parsed = parse_dataset_file(path)
    detection_input = to_feature_space(parsed)

    if not detection_input.mapped_columns:
        raise NoUsableFeaturesError(path, {}, parsed.column_names)

    coverage = feature_coverage(detection_input.records)
    if not any(coverage.values()):
        raise NoUsableFeaturesError(
            path, dict(detection_input.mapped_columns), parsed.column_names
        )

    examples = build_examples(
        detection_input,
        path=path,
        columns=parsed.column_names,
        require_labels=require_labels,
    )
    dataset = validate_examples(examples, feature_names())

    loaded = LoadedFile(
        path=path,
        parsed=parsed,
        mapped_columns=dict(detection_input.mapped_columns),
        dataset=dataset,
        labels=detection_input.labels,
    )
    logger.info(
        "Loaded %s: %d row(s), columns %s.",
        path.name,
        len(dataset),
        ", ".join(f"{k}<-{v}" for k, v in sorted(loaded.mapped_columns.items())),
    )
    return loaded


def load_dataset(
    paths: MLPaths | None = None, *, require_labels: bool = True
) -> LoadedFile:
    """Read every benchmark file in the dataset directory into one dataset.

    Args:
        paths: Directories to read, or ``None`` for the configured ones.
        require_labels: Whether rows without a usable label are refused.

    Returns:
        LoadedFile: The combined rows, with the first file's parsed view.

    Raises:
        DatasetLoadError: If no dataset file is present, or if the combined rows
            fail validation. The message names the directory searched, because
            the usual cause is that no benchmark has been downloaded yet.
    """
    resolved = PATHS if paths is None else paths
    files = discover_dataset_files(resolved)

    if not files:
        raise DatasetLoadError(
            f"No dataset file found in {resolved.dataset_dir}. Place the "
            "benchmark CSVs there; nothing in this package fabricates them."
        )

    loaded = [load_dataset_file(path, require_labels=require_labels) for path in files]
    combined = validate_examples(
        tuple(example for item in loaded for example in item.dataset.examples),
        feature_names(),
    )

    logger.info("Loaded %d dataset file(s) totalling %d row(s).", len(loaded), len(combined))
    return LoadedFile(
        path=loaded[0].path,
        parsed=loaded[0].parsed,
        mapped_columns=loaded[0].mapped_columns,
        dataset=combined,
        labels=tuple(label for item in loaded for label in item.labels),
    )