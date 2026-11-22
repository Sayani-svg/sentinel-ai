"""Reading a real benchmark file into the training feature space.

The generic reader hands a file to :mod:`app.services.log_parser`, the same
parser the ingestion endpoint uses, so an ordinary log upload and a generic
dataset file are read under exactly the same rules, including the parser's row
cap. That cap is correct for a live upload preview, but a CIC-IDS2017 capture has
millions of rows and must not be reduced to the first 100,000 of them. The
``cic_ids2017`` reader therefore uses its own bounded-memory CSV path in
:mod:`src.cic_ids2017_adapter`; the backend parser is left unchanged.

The target format is CICIDS2017 CSVs, which carry a ``Label`` column and the flow
measurements the detector recognises. That is a statement about the label
vocabulary the project already committed to in
:attr:`app.schemas.prediction.AttackType`; it is not a schema invented here.

**Nothing is generated and nothing is dropped.** There is no sample dataset, no
synthetic fallback, and no row fabricated to demonstrate the pipeline. Every
source row is either a training example or an exclusion with a stable reason;
partial loads are reported on the dataset contract and refused by training and
evaluation unless explicitly allowed. A benchmark file must be placed in
``ml/dataset`` by an operator; until then this module raises
:class:`DatasetLoadError` and says where it looked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from app.core.logger import get_logger
from app.services.log_parser import ParsedLog, parse_log_file
from app.services.prediction_service import DetectionInput
from app.services.upload_service import resolve_log_format

from src.cic_ids2017_adapter import (
    CIC_FEATURE_SOURCES,
    CicCsvError,
    CicExclusionReason,
    CicRowExclusion,
    CicStreamLoadResult,
    aggregate_exclusion_counts,
    combine_cic_exclusions,
    load_cic_csv_file,
    validate_cic_examples,
)
from src.config import PATHS, MLPaths
from src.feature_engineering import (
    feature_coverage,
    feature_names,
    to_feature_space,
    to_feature_vector,
)
from src.dataset_validation import (
    DatasetCompleteness,
    DatasetStorage,
    LabeledExample,
    TrainingDataset,
    validate_examples,
)
from src.preprocessing import require_trainable_label

logger = get_logger(__name__)


class DatasetFormat(StrEnum):
    """Dataset readers the loader can select explicitly."""

    GENERIC = "generic"
    CIC_IDS2017 = "cic_ids2017"

#: Extensions the loader will hand to the parser. Exactly the extensions with an
#: implemented reader in :class:`app.schemas.upload.LogFileFormat` -- ``.csv`` and
#: ``.json``. ``.jsonl`` is deliberately absent: the parser has a line-delimited
#: routine internally, but no format value selects it, so offering it here would
#: advertise a loader the application cannot actually use.
SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".csv", ".json"})

#: The CIC reader consumes complete CSV captures only. JSON files in a dataset
#: directory belong to the generic reader and are not CIC sources.
CIC_SUFFIXES: frozenset[str] = frozenset({".csv"})


class DatasetLoadError(RuntimeError):
    """Raised when a dataset file cannot be turned into training rows."""


class UnsupportedDatasetFormatError(DatasetLoadError):
    """Raised when a caller names a dataset reader this build does not have."""

    def __init__(self, value: str) -> None:
        """Record the unsupported value.

        Args:
            value: The reader name the caller supplied.
        """
        supported = ", ".join(member.value for member in DatasetFormat)
        super().__init__(f"Unsupported dataset format {value!r}. Expected one of: {supported}.")


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
        dataset_format: Reader used to produce the rows.
        truncated: Whether any source file exceeded the parser's row cap.
        truncated_paths: Source files whose parse was truncated.
        exclusions: Reason-coded rows dropped by the CIC adapter.
        exclusion_counts: Aggregate count for each CIC exclusion reason.
        rows_seen: Source rows read by the CIC streaming reader; zero for the
            generic reader, which reports retained rows through ``parsed``.
    """

    path: Path
    parsed: ParsedLog
    mapped_columns: dict[str, str]
    dataset: TrainingDataset
    labels: tuple[str, ...]
    dataset_format: DatasetFormat = DatasetFormat.GENERIC
    truncated: bool = False
    truncated_paths: tuple[Path, ...] = ()
    exclusions: Sequence[CicRowExclusion] = ()
    exclusion_counts: Mapping[CicExclusionReason, int] = field(default_factory=dict)
    rows_seen: int = 0


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


def resolve_dataset_format(value: DatasetFormat | str) -> DatasetFormat:
    """Resolve a reader name supplied by a caller.

    Args:
        value: A :class:`DatasetFormat` member or one of its literal values.

    Returns:
        DatasetFormat: The validated reader.

    Raises:
        UnsupportedDatasetFormatError: If the value names no known reader.
    """
    if isinstance(value, DatasetFormat):
        return value
    try:
        return DatasetFormat(value)
    except ValueError as exc:
        raise UnsupportedDatasetFormatError(str(value)) from exc


def _truncation_completeness(path: Path, parsed: ParsedLog) -> DatasetCompleteness:
    """Describe whether the capped generic parser retained every source row.

    Args:
        path: The file that was parsed.
        parsed: Its parsed view, including the parser's truncation flag.

    Returns:
        DatasetCompleteness: A complete record, or one naming the truncated file.
    """
    if not parsed.truncated:
        return DatasetCompleteness()
    return DatasetCompleteness(
        complete=False,
        truncated_paths=(path,),
        reasons=(f"{path.name} exceeded the backend parser row cap and was truncated.",),
    )


def _build_cic_file(
    path: Path,
    result: CicStreamLoadResult,
    *,
    storage: DatasetStorage | str = DatasetStorage.MEMORY,
    storage_dir: Path | None = None,
) -> LoadedFile:
    """Wrap one completely streamed CIC file as a loaded dataset.

    Class coverage is deliberately not checked here: a capture day can hold a
    subset of the eight classes. The exact eight-class check runs after the
    selected files have been combined.

    Args:
        path: The file that was read.
        result: Rows, exclusions and counts from the CIC streaming reader.
        storage: Whether to retain the arrays in memory or on disk.
        storage_dir: Directory for disk-backed arrays, when requested.

    Returns:
        LoadedFile: The five-feature dataset with exclusions and counts preserved.

    Raises:
        DatasetLoadError: If every row was excluded.
        ValueError: If disk storage is requested without a directory.
    """
    if not len(result.dataset):
        counts = ", ".join(
            f"{code.value}={count}" for code, count in result.exclusion_counts.items()
        )
        raise DatasetLoadError(
            f"{path.name} yielded no usable CIC rows. Exclusion counts: "
            f"{counts or 'none recorded'}"
        )
    if result.exclusion_counts:
        counts = ", ".join(
            f"{code.value}={count}"
            for code, count in result.exclusion_counts.items()
        )
        logger.warning("Excluded CIC rows from %s: %s", path.name, counts)

    dataset = result.dataset
    mode = DatasetStorage.parse(storage)
    if mode is DatasetStorage.DISK and dataset.storage is not DatasetStorage.DISK:
        if storage_dir is None:
            raise ValueError("storage_dir is required for disk-backed datasets.")
        dataset = dataset.to_disk(storage_dir, name=f"cic_{path.stem}")
    return LoadedFile(
        path=path,
        parsed=result.parsed,
        mapped_columns=dict(CIC_FEATURE_SOURCES),
        dataset=dataset,
        labels=tuple(label.value for label in dataset.labels),
        dataset_format=DatasetFormat.CIC_IDS2017,
        exclusions=result.exclusions,
        exclusion_counts=dict(result.exclusion_counts),
        rows_seen=result.rows_seen,
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
    path: Path,
    *,
    require_labels: bool = True,
    dataset_format: DatasetFormat | str = DatasetFormat.GENERIC,
    storage: DatasetStorage | str = DatasetStorage.MEMORY,
    storage_dir: Path | None = None,
    exclusions_path: Path | None = None,
) -> LoadedFile:
    """Read one benchmark file and return it as a validated training dataset.

    Args:
        path: The file to read.
        require_labels: Whether rows without a usable label are refused.
        dataset_format: Reader to use. ``generic`` preserves the existing
            production feature mapping; ``cic_ids2017`` uses the adapter that
            derives those same five features from CIC forward/backward columns.
        storage: Whether the validated arrays are resident or memory-mapped.
        storage_dir: Directory for disk-backed arrays, when requested.
        exclusions_path: Optional JSON-lines file for disk-backed CIC exclusions.

    Returns:
        LoadedFile: The parsed file, its column mapping and its validated rows.

    Raises:
        DatasetLoadError: If the file cannot be read, declares none of the
            features this build reads, or carries no usable labels.
        UnsupportedDatasetFormatError: If the named reader does not exist.
        src.dataset_validation.DatasetValidationError: If the rows fail
            validation.
    """
    reader = resolve_dataset_format(dataset_format)
    mode = DatasetStorage.parse(storage)

    if reader is DatasetFormat.CIC_IDS2017:
        if path.suffix.lower() != ".csv":
            raise DatasetLoadError(
                f"{path.name} is not a CSV file; the CIC-IDS2017 reader accepts "
                "complete CSV captures only."
            )
        try:
            result = load_cic_csv_file(
                path,
                storage=mode,
                storage_dir=storage_dir,
                exclusions_path=exclusions_path,
            )
        except CicCsvError as exc:
            raise DatasetLoadError(str(exc)) from exc
        loaded = _build_cic_file(path, result, storage=mode, storage_dir=storage_dir)
    else:
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
        completeness = _truncation_completeness(path, parsed)
        dataset = validate_examples(examples, feature_names(), completeness=completeness)
        if mode is DatasetStorage.DISK:
            if storage_dir is None:
                raise ValueError("storage_dir is required for disk-backed datasets.")
            dataset = dataset.to_disk(storage_dir, name=f"generic_{path.stem}")
        loaded = LoadedFile(
            path=path,
            parsed=parsed,
            mapped_columns=dict(detection_input.mapped_columns),
            dataset=dataset,
            labels=detection_input.labels,
            dataset_format=DatasetFormat.GENERIC,
            truncated=completeness.incomplete,
            truncated_paths=completeness.truncated_paths,
        )

    logger.info(
        "Loaded %s: %d row(s), columns %s.",
        path.name,
        len(loaded.dataset),
        ", ".join(f"{k}<-{v}" for k, v in sorted(loaded.mapped_columns.items())),
    )
    if loaded.truncated:
        logger.warning(
            "Loaded %s is truncated by the parser row cap; its dataset is incomplete.",
            path.name,
        )
    return loaded


def load_dataset(
    paths: MLPaths | None = None,
    *,
    require_labels: bool = True,
    dataset_format: DatasetFormat | str = DatasetFormat.GENERIC,
    files: Sequence[Path] | None = None,
    storage: DatasetStorage | str = DatasetStorage.MEMORY,
    storage_dir: Path | None = None,
    exclusion_dir: Path | None = None,
) -> LoadedFile:
    """Read selected benchmark files into one logically combined dataset.

    Class coverage is a property of the combined dataset. Individual CIC capture
    files may hold any subset of the eight trainable classes; only the union of
    the selected files is held to the exact-eight contract.

    Args:
        paths: Directories to search, or ``None`` for the configured ones.
        require_labels: Whether rows without a usable label are refused.
        dataset_format: Reader to apply to every selected file.
        files: Explicit files to combine, or ``None`` to discover them in
            ``paths.dataset_dir`` in name order. Explicit files are used in the
            order given, so a run over a known set of captures is reproducible.
        storage: Whether combined arrays are resident or memory-mapped.
        storage_dir: Directory for disk-backed arrays, when requested.
        exclusion_dir: Directory for disk-backed row-level CIC exclusions.

    Returns:
        LoadedFile: The combined rows, with aggregate exclusion counts, rows
        seen, and completeness reported across all source files.

    Raises:
        DatasetLoadError: If no dataset file is present, or if the combined rows
            fail validation. The message names the directory searched, because
            the usual cause is that no benchmark has been downloaded yet.
        UnsupportedDatasetFormatError: If the named reader does not exist.
        src.cic_ids2017_adapter.CicClassCoverageError: If the combined CIC rows
            do not cover exactly the eight trainable classes.
    """
    reader = resolve_dataset_format(dataset_format)
    mode = DatasetStorage.parse(storage)
    if mode is DatasetStorage.DISK and storage_dir is None:
        raise ValueError("storage_dir is required for disk-backed datasets.")
    exclusion_target = exclusion_dir or (
        storage_dir / "exclusions" if mode is DatasetStorage.DISK else None
    )
    resolved = PATHS if paths is None else paths
    if files is None:
        discovered = discover_dataset_files(resolved)
        if reader is DatasetFormat.CIC_IDS2017:
            ignored = tuple(
                path for path in discovered if path.suffix.lower() not in CIC_SUFFIXES
            )
            if ignored:
                raise DatasetLoadError(
                    "The CIC-IDS2017 reader will not silently skip non-CSV dataset "
                    "files: " + ", ".join(path.name for path in ignored)
                )
        selected = (
            tuple(path for path in discovered if path.suffix.lower() in CIC_SUFFIXES)
            if reader is DatasetFormat.CIC_IDS2017
            else discovered
        )
    else:
        selected = tuple(files)

    if not selected:
        raise DatasetLoadError(
            f"No dataset file found in {resolved.dataset_dir}. Place the "
            "benchmark CSVs there; nothing in this package fabricates them."
        )

    loaded: list[LoadedFile] = []
    combined: TrainingDataset | None = None
    try:
        for index, path in enumerate(selected):
            exclusion_path = (
                exclusion_target / f"{index}_{path.stem}.exclusions.jsonl"
                if reader is DatasetFormat.CIC_IDS2017 and exclusion_target is not None
                else None
            )
            loaded.append(
                load_dataset_file(
                    path,
                    require_labels=require_labels,
                    dataset_format=reader,
                    storage=mode,
                    storage_dir=storage_dir,
                    exclusions_path=exclusion_path,
                )
            )
        completeness = DatasetCompleteness.merge(
            [item.dataset.completeness for item in loaded]
        )
        combined = TrainingDataset.concat(
            [item.dataset for item in loaded],
            storage=mode,
            storage_dir=storage_dir,
            name="combined",
            completeness=completeness,
        )
        if reader is DatasetFormat.CIC_IDS2017:
            validate_cic_examples(combined.examples)

        if reader is DatasetFormat.CIC_IDS2017 and exclusion_target is not None:
            combined_exclusion_path = exclusion_target / "combined.exclusions.jsonl"
            exclusions: Sequence[CicRowExclusion] = combine_cic_exclusions(
                [item.exclusions for item in loaded],
                combined_exclusion_path,
            )
            combined = replace(
                combined,
                cleanup_paths=tuple(
                    dict.fromkeys((*combined.cleanup_paths, combined_exclusion_path))
                ),
            )
        else:
            exclusions = tuple(
                exclusion for item in loaded for exclusion in item.exclusions
            )
        truncated_paths = tuple(item.path for item in loaded if item.truncated)
        logger.info(
            "Loaded %d dataset file(s) totalling %d row(s) from %d source row(s).",
            len(loaded),
            len(combined),
            sum(item.rows_seen for item in loaded),
        )
        if truncated_paths:
            logger.warning(
                "Dataset load is incomplete; parser truncation affected: %s",
                ", ".join(path.name for path in truncated_paths),
            )
        return LoadedFile(
            path=loaded[0].path,
            parsed=loaded[0].parsed,
            mapped_columns=loaded[0].mapped_columns,
            dataset=combined,
            labels=tuple(label for item in loaded for label in item.labels),
            dataset_format=reader,
            truncated=completeness.incomplete,
            truncated_paths=truncated_paths,
            exclusions=exclusions,
            exclusion_counts=aggregate_exclusion_counts(exclusions),
            rows_seen=sum(item.rows_seen for item in loaded),
        )
    except BaseException:
        for item in loaded:
            item.dataset.cleanup()
        if combined is not None:
            combined.cleanup()
        raise
