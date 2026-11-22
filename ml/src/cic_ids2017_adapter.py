"""CIC-IDS2017 adapter for the Sentinel training feature space.

The adapter isolates CIC-specific column names, label spellings and feature
derivation. It produces vectors in the canonical order recorded by
:func:`src.feature_engineering.feature_names` and never changes the generic
features used by production inference.

Rows that cannot be used are excluded rather than imputed. Every exclusion
carries a stable reason code so a dataset report can explain exactly what was
dropped and why.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Final, NoReturn

import numpy as np
from app.schemas.prediction import AttackType
from app.schemas.upload import LogFileFormat, LogValue, LogValueType
from app.services.log_parser import (
    LogColumn,
    ParsedLog,
    coerce_scalar,
)

from src.dataset_validation import (
    DatasetStorage,
    LabeledExample,
    TrainingDataset,
    close_memmap,
    validate_arrays,
)
from src.feature_engineering import feature_names
from src.preprocessing import (
    CLASS_COUNT,
    TRAINABLE_ATTACK_TYPES,
    label_to_index,
)

CIC_EXPECTED_CLASS_COUNT: Final[int] = 8

CIC_EXPECTED_FEATURES: Final[tuple[str, ...]] = (
    "flow_duration",
    "total_bytes",
    "total_packets",
    "avg_packet_size",
    "packet_rate",
)

CIC_LABEL_TO_TRAINABLE: Final[dict[str, AttackType]] = {
    "benign": AttackType.BENIGN,
    "dos hulk": AttackType.DOS,
    "dos goldeneye": AttackType.DOS,
    "goldeneye": AttackType.DOS,
    "dos slowloris": AttackType.DOS,
    "slowloris": AttackType.DOS,
    "dos slowhttptest": AttackType.DOS,
    "slowhttptest": AttackType.DOS,
    "ddos": AttackType.DDOS,
    "portscan": AttackType.PORT_SCAN,
    "port scan": AttackType.PORT_SCAN,
    "bot": AttackType.BOT,
    "botnet ares": AttackType.BOT,
    "brute force": AttackType.BRUTE_FORCE,
    "ftp patator": AttackType.BRUTE_FORCE,
    "ssh patator": AttackType.BRUTE_FORCE,
    "web attack brute force": AttackType.BRUTE_FORCE,
    "web attack": AttackType.WEB_ATTACK,
    "web attack xss": AttackType.WEB_ATTACK,
    "xss": AttackType.WEB_ATTACK,
    "sql injection": AttackType.WEB_ATTACK,
    "web attack sql injection": AttackType.WEB_ATTACK,
    "infiltration": AttackType.INFILTRATION,
}

EXCLUDED_CIC_LABEL_FRAGMENTS: Final[tuple[str, ...]] = ("heartbleed",)

CIC_CHUNK_SIZE: Final[int] = 10_000

_INTEGER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[+-]?[0-9]+")
_FLOAT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?"
)
_NON_FINITE_SPELLINGS: Final[frozenset[str]] = frozenset(
    {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }
)

CIC_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "flow_duration": ("Flow Duration", "flow_duration"),
    "total_fwd_packets": ("Total Fwd Packets", "Fwd Packets"),
    "total_bwd_packets": ("Total Backward Packets", "Bwd Packets"),
    "total_fwd_bytes": ("Total Length of Fwd Packets", "Fwd Packet Length Total"),
    "total_bwd_bytes": ("Total Length of Bwd Packets", "Bwd Packet Length Total"),
    "avg_packet_size": ("Average Packet Size", "Avg Packet Size"),
    "flow_packets_s": ("Flow Packets/s", "Flow Packets per Second"),
    "label": ("Label", "label"),
}

CIC_FEATURE_SOURCES: Final[dict[str, str]] = {
    "flow_duration": "Flow Duration",
    "total_bytes": "Total Length of Fwd Packets + Total Length of Bwd Packets",
    "total_packets": "Total Fwd Packets + Total Backward Packets",
    "avg_packet_size": "Average Packet Size",
    "packet_rate": "Flow Packets/s",
    "label": "Label",
}


class CicAdapterError(ValueError):
    """Base class for adapter contract and row-level failures."""


class CicFeatureContractError(CicAdapterError):
    """Raised when the canonical feature set is not the expected five features."""


class CicClassCoverageError(CicAdapterError):
    """Raised when a CIC dataset does not cover all eight trainable classes."""

    def __init__(
        self,
        *,
        missing: Sequence[AttackType],
        extra: Sequence[AttackType] = (),
        path: Path | None = None,
    ) -> None:
        """Record the class-coverage gap.

        Args:
            missing: Trainable classes absent from the dataset.
            extra: Classes outside the trainable vocabulary that were present.
            path: Dataset file whose coverage failed, when known.
        """
        location = f" in {path.name}" if path is not None else ""
        expected = ", ".join(a.value for a in TRAINABLE_ATTACK_TYPES)
        missing_text = ", ".join(a.value for a in missing) or "none"
        extra_text = ", ".join(a.value for a in extra) or "none"
        super().__init__(
            f"CIC dataset{location} must cover exactly {CIC_EXPECTED_CLASS_COUNT} "
            f"trainable Sentinel classes: {expected}. Missing: {missing_text}. "
            f"Unexpected: {extra_text}."
        )
        self.missing = tuple(missing)
        self.extra = tuple(extra)
        self.path = path


class CicCsvError(CicAdapterError):
    """Raised when a CIC CSV file is structurally unreadable."""


class CicExclusionReason(StrEnum):
    """Stable reason codes for rows excluded by the adapter."""

    MISSING_LABEL = "missing_label"
    HEARTBLEED = "heartbleed_excluded"
    UNKNOWN_LABEL = "unknown_label"
    MISSING_VALUE = "missing_value"
    UNPARSABLE_VALUE = "unparsable_value"
    NON_FINITE_VALUE = "non_finite_value"
    NON_FINITE_DERIVED_VALUE = "non_finite_derived_value"


class CicRowExcludedError(CicAdapterError):
    """Raised for one CIC row that cannot become a training example."""

    def __init__(
        self,
        row_index: int,
        reason: str,
        code: CicExclusionReason = CicExclusionReason.UNPARSABLE_VALUE,
    ) -> None:
        """Record the row position and the reason it was excluded.

        Args:
            row_index: Zero-based position of the row in the parsed file.
            reason: Human-readable explanation of the exclusion.
            code: Stable reason code assigned to the exclusion.
        """
        super().__init__(f"Row {row_index} excluded [{code.value}]: {reason}")
        self.row_index = row_index
        self.reason = reason
        self.code = code


@dataclass(frozen=True, slots=True)
class CicRowExclusion:
    """One retained exclusion record from a CIC conversion."""

    row_index: int
    code: CicExclusionReason
    reason: str


@dataclass(frozen=True, slots=True)
class CicLabelResolution:
    """Result of resolving a raw CIC label against the trainable vocabulary."""

    normalized: str
    attack_type: AttackType | None
    code: CicExclusionReason | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class CicConversionResult:
    """Usable rows and exclusions produced from one CIC source."""

    examples: tuple[LabeledExample, ...]
    exclusions: tuple[CicRowExclusion, ...]


@dataclass(frozen=True, slots=True)
class CicStreamLoadResult:
    """Rows, exclusions and counts from streaming one CIC CSV file."""

    dataset: TrainingDataset
    exclusions: Sequence[CicRowExclusion]
    exclusion_counts: Mapping[CicExclusionReason, int]
    rows_seen: int
    parsed: ParsedLog

    @property
    def examples(self) -> Sequence[LabeledExample]:
        """Return the loaded rows as a lazy compatibility view."""
        return self.dataset.examples


def aggregate_exclusion_counts(
    exclusions: Sequence[CicRowExclusion],
) -> dict[CicExclusionReason, int]:
    """Count exclusions by stable reason code.

    Args:
        exclusions: Row-level exclusion records, from one file or many.

    Returns:
        dict[CicExclusionReason, int]: Reason code to number of excluded rows.
        Reasons with no exclusions are absent rather than reported as zero.
    """
    return {
        code: count
        for code, count in sorted(
            Counter(item.code for item in exclusions).items(),
            key=lambda item: item[0].value,
        )
    }


def _safe_stem(path: Path) -> str:
    """Return a filename-safe dataset stem for storage artifacts."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem) or "cic_dataset"


def _open_temporary_file(path: Path) -> tuple[Path, BinaryIO]:
    """Open a temporary sibling file for a storage artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(temporary_name), Path(temporary_name).open("wb")


class _ArrayAccumulator:
    """Collect validated rows as numeric arrays, optionally on disk."""

    def __init__(
        self,
        width: int,
        storage: DatasetStorage,
        directory: Path | None,
        name: str,
    ) -> None:
        self._width = width
        self._storage = storage
        self._rows = 0
        self._feature_chunks: list[np.ndarray] = []
        self._code_chunks: list[np.ndarray] = []
        self._feature_path: Path | None = None
        self._code_path: Path | None = None
        self._feature_temp_path: Path | None = None
        self._code_temp_path: Path | None = None
        self._feature_handle: BinaryIO | None = None
        self._code_handle: BinaryIO | None = None
        self._final_paths: tuple[Path, Path] | None = None
        self._finalized = False
        if storage is DatasetStorage.DISK:
            if directory is None:
                raise ValueError("storage_dir is required for disk-backed CIC loads.")
            self._feature_path = directory / f"{name}.features.dat"
            self._code_path = directory / f"{name}.labels.dat"
            try:
                self._feature_temp_path, self._feature_handle = _open_temporary_file(
                    self._feature_path
                )
                self._code_temp_path, self._code_handle = _open_temporary_file(
                    self._code_path
                )
            except BaseException:
                self.abort()
                raise

    def append(self, features: np.ndarray, codes: np.ndarray) -> None:
        """Append one chunk of already-converted rows."""
        if not features.shape[0]:
            return
        if self._storage is DatasetStorage.DISK:
            assert self._feature_handle is not None
            assert self._code_handle is not None
            features.tofile(self._feature_handle)
            codes.tofile(self._code_handle)
        else:
            self._feature_chunks.append(features)
            self._code_chunks.append(codes)
        self._rows += int(features.shape[0])

    def finish(self) -> tuple[np.ndarray, np.ndarray, tuple[Path, ...]]:
        """Return the accumulated arrays, publishing disk files atomically."""
        if self._storage is DatasetStorage.DISK:
            try:
                for handle in (self._feature_handle, self._code_handle):
                    if handle is not None and not handle.closed:
                        handle.close()
                self._feature_handle = None
                self._code_handle = None
                if self._rows == 0:
                    self.abort()
                    return (
                        np.empty((0, self._width), dtype=np.float32),
                        np.empty(0, dtype=np.uint8),
                        (),
                    )
                assert self._feature_path is not None
                assert self._code_path is not None
                assert self._feature_temp_path is not None
                assert self._code_temp_path is not None
                self._final_paths = (self._feature_path, self._code_path)
                self._finalized = True
                os.replace(self._feature_temp_path, self._feature_path)
                os.replace(self._code_temp_path, self._code_path)
                features = np.memmap(
                    self._feature_path,
                    dtype=np.float32,
                    mode="r",
                    shape=(self._rows, self._width),
                )
                codes = np.memmap(
                    self._code_path,
                    dtype=np.uint8,
                    mode="r",
                    shape=(self._rows,),
                )
                return features, codes, self._final_paths
            except BaseException:
                self.abort()
                raise
        if self._feature_chunks:
            features = np.concatenate(self._feature_chunks)
            codes = np.concatenate(self._code_chunks)
        else:
            features = np.empty((0, self._width), dtype=np.float32)
            codes = np.empty(0, dtype=np.uint8)
        return features, codes, ()

    def abort(self) -> None:
        """Close open writers and remove every file this accumulator created."""
        for handle in (self._feature_handle, self._code_handle):
            if handle is not None and not handle.closed:
                handle.close()
        self._feature_handle = None
        self._code_handle = None
        for path in (self._feature_temp_path, self._code_temp_path):
            if path is not None:
                path.unlink(missing_ok=True)
        if self._finalized and self._final_paths is not None:
            for path in self._final_paths:
                path.unlink(missing_ok=True)
        self._feature_temp_path = None
        self._code_temp_path = None
        self._final_paths = None
        self._finalized = False

    @property
    def rows(self) -> int:
        """Return how many usable rows have been appended."""
        return self._rows


def _decode_exclusion(payload: Mapping[str, object]) -> CicRowExclusion:
    """Rebuild one exclusion record from its JSON-line spelling."""
    return CicRowExclusion(
        row_index=int(payload["row_index"]),
        code=CicExclusionReason(str(payload["code"])),
        reason=str(payload["reason"]),
    )


class _CicExclusionFile(Sequence[CicRowExclusion]):
    """A disk-backed, read-only sequence of row-level exclusions."""

    __slots__ = ("_path", "_count")

    def __init__(self, path: Path, count: int) -> None:
        self._path = path
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[CicRowExclusion]:
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield _decode_exclusion(json.loads(line))

    def __getitem__(
        self, index: int | slice
    ) -> CicRowExclusion | tuple[CicRowExclusion, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        position = index + len(self) if index < 0 else index
        if not 0 <= position < len(self):
            raise IndexError("exclusion index out of range")
        for offset, item in enumerate(self):
            if offset == position:
                return item
        raise IndexError("exclusion index out of range")

    def __repr__(self) -> str:
        return f"CicExclusionFile(rows={self._count}, path={str(self._path)!r})"


class _CicExclusionWriter:
    """Write exclusions as JSON lines and expose them as a lazy sequence."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._temporary_path: Path | None = None
        self._handle: BinaryIO | None = None
        self._count = 0
        self._finalized = False
        try:
            self._temporary_path, self._handle = _open_temporary_file(path)
        except BaseException:
            self.abort()
            raise

    def append(self, exclusion: CicRowExclusion) -> None:
        """Persist one exclusion record."""
        assert self._handle is not None
        payload = {
            "row_index": exclusion.row_index,
            "code": exclusion.code.value,
            "reason": exclusion.reason,
        }
        line = json.dumps(payload, ensure_ascii=True) + "\n"
        self._handle.write(line.encode("utf-8"))
        self._count += 1

    def close(self) -> None:
        """Close the writer without publishing the file."""
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        self._handle = None

    def finish(self) -> _CicExclusionFile:
        """Publish the JSONL file atomically and return a lazy reader."""
        try:
            self.close()
            assert self._temporary_path is not None
            os.replace(self._temporary_path, self._path)
            self._temporary_path = None
            self._finalized = True
            return _CicExclusionFile(self._path, self._count)
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        """Close the writer and remove temporary or published files."""
        self.close()
        if self._temporary_path is not None:
            self._temporary_path.unlink(missing_ok=True)
            self._temporary_path = None
        if self._finalized:
            self._path.unlink(missing_ok=True)
            self._finalized = False


def _sorted_exclusion_counts(
    counts: Mapping[CicExclusionReason, int],
) -> dict[CicExclusionReason, int]:
    """Return reason counts ordered by stable reason code."""
    return {
        code: count
        for code, count in sorted(counts.items(), key=lambda item: item[0].value)
    }


def combine_cic_exclusions(
    exclusions: Sequence[Sequence[CicRowExclusion]], path: Path
) -> Sequence[CicRowExclusion]:
    """Write several per-file exclusion sequences into one disk-backed store."""
    writer = _CicExclusionWriter(path)
    try:
        for source in exclusions:
            for item in source:
                writer.append(item)
        return writer.finish()
    except BaseException:
        writer.abort()
        raise


def assert_cic_contract() -> None:
    """Verify that the adapter and production feature/class contracts agree.

    Raises:
        CicFeatureContractError: If the canonical feature vector is not the
            expected five production features in the expected order.
        CicClassCoverageError: If the trainable vocabulary is not exactly eight
            classes or the CIC label map does not cover all of them.
    """
    if tuple(feature_names()) != CIC_EXPECTED_FEATURES:
        raise CicFeatureContractError(
            f"CIC adapter expects features {CIC_EXPECTED_FEATURES}, but production "
            f"defines {tuple(feature_names())}."
        )
    if CLASS_COUNT != CIC_EXPECTED_CLASS_COUNT:
        raise CicClassCoverageError(missing=())
    mapped = set(CIC_LABEL_TO_TRAINABLE.values())
    expected = set(TRAINABLE_ATTACK_TYPES)
    if mapped != expected:
        raise CicClassCoverageError(
            missing=tuple(a for a in TRAINABLE_ATTACK_TYPES if a not in mapped),
            extra=tuple(a for a in mapped if a not in expected),
        )


def normalize_cic_text(value: object) -> str:
    """Normalize CIC text for case-insensitive, whitespace-tolerant matching.

    Hyphens, en dashes, em dashes and underscores are treated as whitespace so
    spellings such as ``Web Attack - XSS`` and ``web attack-xss`` resolve to the
    same label.

    Args:
        value: Raw label value from a dataset row.

    Returns:
        str: Lower-case, whitespace-collapsed text, or an empty string.
    """
    if value is None:
        return ""
    text = str(value)
    for separator in ("-", "\u2013", "\u2014", "_"):
        text = text.replace(separator, " ")
    return " ".join(text.split()).lower()


def resolve_cic_label(label: object) -> CicLabelResolution:
    """Resolve a raw CIC label to a trainable class or an explicit exclusion.

    Args:
        label: Raw label value from a dataset row.

    Returns:
        CicLabelResolution: The resolved class, or the reason it was excluded.
    """
    normalized = normalize_cic_text(label)
    if not normalized:
        return CicLabelResolution(
            normalized=normalized,
            attack_type=None,
            code=CicExclusionReason.MISSING_LABEL,
            reason="the Label cell is missing or blank",
        )

    compact = normalized.replace(" ", "")
    if any(
        fragment in normalized or fragment in compact
        for fragment in EXCLUDED_CIC_LABEL_FRAGMENTS
    ):
        return CicLabelResolution(
            normalized=normalized,
            attack_type=None,
            code=CicExclusionReason.HEARTBLEED,
            reason=(
                f"label {normalized!r} is Heartbleed, which is excluded from the "
                "eight-class Sentinel contract"
            ),
        )

    attack_type = CIC_LABEL_TO_TRAINABLE.get(normalized)
    if attack_type is None:
        return CicLabelResolution(
            normalized=normalized,
            attack_type=None,
            code=CicExclusionReason.UNKNOWN_LABEL,
            reason=f"label {normalized!r} does not map to a trainable Sentinel class",
        )

    return CicLabelResolution(
        normalized=normalized,
        attack_type=attack_type,
        code=None,
        reason=None,
    )


def normalize_cic_label_to_trainable(label: object) -> AttackType | None:
    """Resolve a CIC label while preserving the original boolean-style API.

    Args:
        label: Raw label value from a dataset row.

    Returns:
        AttackType | None: The trainable class, or ``None`` when excluded.
    """
    return resolve_cic_label(label).attack_type


def _get_col(row: Mapping[str, object], candidates: tuple[str, ...]) -> object | None:
    """Find a source column by exact, case-insensitive or contained header.

    Args:
        row: One parsed dataset row.
        candidates: Accepted header spellings for the source measurement.

    Returns:
        object | None: The matching cell value, or ``None`` when absent.
    """
    for key in row.keys():
        name = str(key).strip()
        for candidate in candidates:
            if name == candidate or name.lower() == candidate.lower():
                return row[key]
    for key in row.keys():
        name = str(key).strip().lower()
        for candidate in candidates:
            if candidate.lower() in name:
                return row[key]
    return None


def _raise_row_exclusion(
    row_index: int, code: CicExclusionReason, message: str
) -> NoReturn:
    """Raise a reason-coded row exclusion.

    Args:
        row_index: Zero-based position of the excluded row.
        code: Stable reason code.
        message: Human-readable explanation.

    Raises:
        CicRowExcludedError: Always.
    """
    raise CicRowExcludedError(row_index, message, code)


def _to_float(value: object, row_index: int, feature: str) -> float:
    """Coerce one CIC measurement to a finite float or exclude the row.

    Accepted strings use the backend's numeric grammar: optional sign, decimal
    or scientific notation, and no Python-specific separators such as ``1_0``.
    The patterns below mirror :mod:`app.services.log_parser`, which keeps such a
    spelling as text; a CIC row must never turn it into ``10.0``.

    Args:
        value: Raw cell value from the parsed row.
        row_index: Zero-based position of the row.
        feature: Canonical name used in the exclusion reason.

    Returns:
        float: The finite numeric value.

    Raises:
        CicRowExcludedError: If the value is missing, non-numeric, boolean or
            non-finite.
    """
    if value is None:
        _raise_row_exclusion(
            row_index,
            CicExclusionReason.MISSING_VALUE,
            f"{feature} is missing",
        )
    if isinstance(value, bool):
        _raise_row_exclusion(
            row_index,
            CicExclusionReason.UNPARSABLE_VALUE,
            f"{feature} is boolean, which is not a numeric measurement",
        )
    if isinstance(value, str):
        text = value.strip()
        if not text:
            _raise_row_exclusion(
                row_index,
                CicExclusionReason.MISSING_VALUE,
                f"{feature} is blank",
            )
        if text.lower() in _NON_FINITE_SPELLINGS:
            _raise_row_exclusion(
                row_index,
                CicExclusionReason.NON_FINITE_VALUE,
                f"{feature} value {value!r} is non-finite",
            )
        if not _INTEGER_PATTERN.fullmatch(text) and not _FLOAT_PATTERN.fullmatch(text):
            _raise_row_exclusion(
                row_index,
                CicExclusionReason.UNPARSABLE_VALUE,
                f"{feature} value {value!r} is not numeric",
            )
        number = float(text)
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        _raise_row_exclusion(
            row_index,
            CicExclusionReason.UNPARSABLE_VALUE,
            f"{feature} value {value!r} is not numeric",
        )

    if not math.isfinite(number):
        _raise_row_exclusion(
            row_index,
            CicExclusionReason.NON_FINITE_VALUE,
            f"{feature} value {value!r} is non-finite",
        )
    return number


def derive_cic_features(row: Mapping[str, object], row_index: int) -> tuple[float, ...]:
    """Derive the canonical five-feature vector from one CIC row.

    Args:
        row: One parsed CIC dataset row.
        row_index: Zero-based position of the row.

    Returns:
        tuple[float, ...]: Five floats in production feature order.

    Raises:
        CicFeatureContractError: If production features changed.
        CicRowExcludedError: If a source value is missing or invalid, or if a
            derived value is non-finite.
    """
    assert_cic_contract()
    flow_duration = _to_float(
        _get_col(row, CIC_COLUMNS["flow_duration"]), row_index, "flow_duration"
    )
    fwd_packets = _to_float(
        _get_col(row, CIC_COLUMNS["total_fwd_packets"]), row_index, "total_fwd_packets"
    )
    bwd_packets = _to_float(
        _get_col(row, CIC_COLUMNS["total_bwd_packets"]), row_index, "total_bwd_packets"
    )
    fwd_bytes = _to_float(
        _get_col(row, CIC_COLUMNS["total_fwd_bytes"]), row_index, "total_fwd_bytes"
    )
    bwd_bytes = _to_float(
        _get_col(row, CIC_COLUMNS["total_bwd_bytes"]), row_index, "total_bwd_bytes"
    )
    avg_packet_size = _to_float(
        _get_col(row, CIC_COLUMNS["avg_packet_size"]), row_index, "avg_packet_size"
    )
    packet_rate = _to_float(
        _get_col(row, CIC_COLUMNS["flow_packets_s"]), row_index, "flow_packets_s"
    )

    total_bytes = fwd_bytes + bwd_bytes
    total_packets = fwd_packets + bwd_packets
    if not math.isfinite(total_bytes) or not math.isfinite(total_packets):
        _raise_row_exclusion(
            row_index,
            CicExclusionReason.NON_FINITE_DERIVED_VALUE,
            "derived total_bytes or total_packets is non-finite",
        )

    values = {
        "flow_duration": flow_duration,
        "total_bytes": total_bytes,
        "total_packets": total_packets,
        "avg_packet_size": avg_packet_size,
        "packet_rate": packet_rate,
    }
    return tuple(values[name] for name in feature_names())


def cic_row_to_example(
    row: Mapping[str, object], row_index: int
) -> tuple[tuple[float, ...], AttackType]:
    """Convert one CIC row into a feature vector and trainable label.

    Args:
        row: One parsed CIC dataset row.
        row_index: Zero-based position of the row.

    Returns:
        tuple[tuple[float, ...], AttackType]: The five-feature vector and class.

    Raises:
        CicFeatureContractError: If production features changed.
        CicRowExcludedError: If the label or any numeric value cannot be used.
    """
    assert_cic_contract()
    resolution = resolve_cic_label(_get_col(row, CIC_COLUMNS["label"]))
    if resolution.attack_type is None:
        _raise_row_exclusion(
            row_index,
            resolution.code or CicExclusionReason.UNKNOWN_LABEL,
            resolution.reason or "label is unusable",
        )
    attack_type = resolution.attack_type
    return derive_cic_features(row, row_index), attack_type


def convert_cic_rows(rows: Sequence[Mapping[str, object]]) -> CicConversionResult:
    """Convert CIC rows into validated examples and explicit exclusion records.

    Args:
        rows: Parsed dataset rows in file order.

    Returns:
        CicConversionResult: Usable examples and reason-coded exclusions.
    """
    assert_cic_contract()
    examples: list[LabeledExample] = []
    exclusions: list[CicRowExclusion] = []
    for row_index, row in enumerate(rows):
        try:
            features, label = cic_row_to_example(row, row_index)
        except CicRowExcludedError as exc:
            exclusions.append(
                CicRowExclusion(
                    row_index=exc.row_index,
                    code=exc.code,
                    reason=exc.reason,
                )
            )
            continue
        examples.append(LabeledExample(features=features, label=label))
    return CicConversionResult(
        examples=tuple(examples),
        exclusions=tuple(exclusions),
    )


def _validate_cic_header(header: list[str]) -> tuple[str, ...]:
    """Validate a CIC CSV header without using the capped backend parser."""
    names: list[str] = []
    seen: set[str] = set()
    for index, raw_name in enumerate(header):
        name = raw_name.strip()
        if not name:
            raise CicCsvError(f"Column {index + 1} of the header row has no name.")
        if name in seen:
            raise CicCsvError(f"Column name {name!r} appears more than once.")
        seen.add(name)
        names.append(name)
    return tuple(names)


def iter_cic_csv_chunks(
    path: Path, *, chunk_size: int = CIC_CHUNK_SIZE
) -> Iterator[tuple[tuple[str, ...], tuple[Mapping[str, LogValue], ...]]]:
    """Yield fixed-size chunks of CIC CSV rows from a streaming reader.

    The whole file is never read into memory. Only one chunk of parsed records is
    held at a time, and there is no row cap: every non-blank data row is yielded.

    Args:
        path: CIC CSV file to read.
        chunk_size: Maximum parsed rows held per chunk.

    Yields:
        tuple: Header names and one chunk of parsed records.

    Raises:
        CicCsvError: If the file is not CSV, is empty, or has a malformed row.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if path.suffix.lower() != ".csv":
        raise CicCsvError(f"{path.name} is not a CIC CSV file.")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header: tuple[str, ...] | None = None
        for candidate in reader:
            if any(field.strip() for field in candidate):
                header = _validate_cic_header(candidate)
                break
        if header is None:
            raise CicCsvError(f"{path.name} does not contain a header row.")

        expected_width = len(header)
        chunk: list[Mapping[str, LogValue]] = []
        for raw_row in reader:
            if not raw_row or not any(field.strip() for field in raw_row):
                continue
            if len(raw_row) != expected_width:
                raise CicCsvError(
                    f"{path.name} line {reader.line_num} has {len(raw_row)} fields "
                    f"but the header declares {expected_width}."
                )
            chunk.append(
                {
                    name: coerce_scalar(value)
                    for name, value in zip(header, raw_row, strict=True)
                }
            )
            if len(chunk) >= chunk_size:
                yield header, tuple(chunk)
                chunk.clear()
        if chunk:
            yield header, tuple(chunk)


def load_cic_csv_file(
    path: Path,
    *,
    chunk_size: int = CIC_CHUNK_SIZE,
    storage: DatasetStorage | str = DatasetStorage.MEMORY,
    storage_dir: Path | None = None,
    exclusions_path: Path | None = None,
) -> CicStreamLoadResult:
    """Convert a complete CIC CSV file with bounded raw-row memory.

    Rows become compact ``float32``/``uint8`` arrays instead of Python objects.
    With ``storage=DISK`` the arrays are appended straight to memory-mapped
    files; with ``exclusions_path`` the row-level exclusions are written as JSON
    lines instead of being retained as objects.

    Args:
        path: CIC CSV file to read.
        chunk_size: Maximum parsed rows held during ingestion.
        storage: Whether usable rows are kept in memory or memory-mapped.
        storage_dir: Directory for disk-backed arrays, when requested.
        exclusions_path: Optional JSON-lines file for disk-backed exclusions.

    Returns:
        CicStreamLoadResult: Validated rows, exclusions, aggregate counts and a
        row-free parsed view of the file.

    Raises:
        CicCsvError: If the file has no data rows or no usable rows.
        ValueError: If disk storage is requested without a directory.
    """
    assert_cic_contract()
    mode = DatasetStorage.parse(storage)
    width = len(CIC_EXPECTED_FEATURES)
    accumulator: _ArrayAccumulator | None = None
    writer: _CicExclusionWriter | None = None
    open_arrays: list[np.ndarray] = []
    try:
        accumulator = _ArrayAccumulator(width, mode, storage_dir, _safe_stem(path))
        writer = _CicExclusionWriter(exclusions_path) if exclusions_path else None
        memory_exclusions: list[CicRowExclusion] = []
        counts: Counter[CicExclusionReason] = Counter()
        rows_seen = 0
        columns: tuple[LogColumn, ...] | None = None

        def record_exclusion(exc: CicRowExcludedError) -> None:
            exclusion = CicRowExclusion(
                row_index=exc.row_index,
                code=exc.code,
                reason=exc.reason,
            )
            counts[exclusion.code] += 1
            if writer is None:
                memory_exclusions.append(exclusion)
            else:
                writer.append(exclusion)

        for header, chunk in iter_cic_csv_chunks(path, chunk_size=chunk_size):
            if columns is None:
                columns = tuple(
                    LogColumn(name=name, value_type=LogValueType.STRING, null_count=0)
                    for name in header
                )
            usable_features = np.empty((len(chunk), width), dtype=np.float32)
            usable_codes = np.empty(len(chunk), dtype=np.uint8)
            usable = 0
            for row_index, row in enumerate(chunk, start=rows_seen):
                rows_seen += 1
                try:
                    features, label = cic_row_to_example(row, row_index)
                except CicRowExcludedError as exc:
                    record_exclusion(exc)
                    continue
                usable_features[usable] = features
                usable_codes[usable] = label_to_index(label)
                usable += 1
            accumulator.append(usable_features[:usable], usable_codes[:usable])

        if rows_seen == 0:
            raise CicCsvError(f"{path.name} contains a header row but no data rows.")
        if accumulator.rows == 0:
            counts_text = ", ".join(
                f"{code.value}={count}"
                for code, count in _sorted_exclusion_counts(counts).items()
            )
            raise CicCsvError(
                f"{path.name} yielded no usable CIC rows. Exclusion counts: "
                f"{counts_text or 'none recorded'}"
            )

        features, codes, storage_paths = accumulator.finish()
        open_arrays.extend((features, codes))
        dataset = validate_arrays(
            features,
            codes,
            CIC_EXPECTED_FEATURES,
            require_class_coverage=False,
            copy_arrays=False,
        )
        cleanup_paths = tuple(
            dict.fromkeys(
                (
                    *storage_paths,
                    *((exclusions_path,) if exclusions_path is not None else ()),
                )
            )
        )
        if storage_paths or cleanup_paths:
            dataset = replace(
                dataset,
                storage=mode,
                storage_paths=storage_paths,
                cleanup_paths=cleanup_paths,
            )
        parsed = ParsedLog(
            file_format=LogFileFormat.CSV,
            columns=columns or (),
            rows=(),
            truncated=False,
        )
        return CicStreamLoadResult(
            dataset=dataset,
            exclusions=writer.finish() if writer else tuple(memory_exclusions),
            exclusion_counts=_sorted_exclusion_counts(counts),
            rows_seen=rows_seen,
            parsed=parsed,
        )
    except BaseException:
        for array in open_arrays:
            close_memmap(array)
        if accumulator is not None:
            accumulator.abort()
        if writer is not None:
            writer.abort()
        raise


def validate_cic_examples(
    examples: Sequence[LabeledExample], *, path: Path | None = None
) -> None:
    """Require exact coverage of the eight trainable Sentinel classes.

    Args:
        examples: Validated training rows produced from a CIC source.
        path: Dataset file whose coverage is being checked, when known.

    Raises:
        CicFeatureContractError: If the production feature contract changed.
        CicClassCoverageError: If a trainable class is missing or an unexpected
            class is present.
    """
    assert_cic_contract()
    expected = set(TRAINABLE_ATTACK_TYPES)
    present = {example.label for example in examples}
    missing = tuple(a for a in TRAINABLE_ATTACK_TYPES if a not in present)
    extra = tuple(a for a in present if a not in expected)
    if missing or extra:
        raise CicClassCoverageError(missing=missing, extra=extra, path=path)


__all__ = [
    "CIC_CHUNK_SIZE",
    "CIC_EXPECTED_CLASS_COUNT",
    "CIC_EXPECTED_FEATURES",
    "CIC_FEATURE_SOURCES",
    "CIC_LABEL_TO_TRAINABLE",
    "CicAdapterError",
    "CicClassCoverageError",
    "CicConversionResult",
    "CicCsvError",
    "CicExclusionReason",
    "CicFeatureContractError",
    "CicLabelResolution",
    "CicRowExcludedError",
    "CicRowExclusion",
    "CicStreamLoadResult",
    "aggregate_exclusion_counts",
    "assert_cic_contract",
    "combine_cic_exclusions",
    "convert_cic_rows",
    "cic_row_to_example",
    "derive_cic_features",
    "iter_cic_csv_chunks",
    "load_cic_csv_file",
    "normalize_cic_label_to_trainable",
    "normalize_cic_text",
    "resolve_cic_label",
    "validate_cic_examples",
]
