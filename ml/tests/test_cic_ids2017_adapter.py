"""Focused tests for CIC-IDS2017 ingestion, class coverage and row contracts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Final, Mapping

import pytest
from app.schemas.prediction import AttackType

import app.services.log_parser as log_parser
import src.cic_ids2017_adapter as cic_adapter
import src.data_loader as data_loader
from src.cic_ids2017_adapter import (
    CIC_EXPECTED_FEATURES,
    CIC_FEATURE_SOURCES,
    CIC_LABEL_TO_TRAINABLE,
    CicClassCoverageError,
    CicExclusionReason,
    CicRowExcludedError,
    assert_cic_contract,
    cic_row_to_example,
    normalize_cic_label_to_trainable,
)
from src.config import DEFAULT_MODEL, MLPaths
from src.dataset_validation import DatasetStorage
from src.data_loader import (
    DatasetFormat,
    DatasetLoadError,
    UnsupportedDatasetFormatError,
    load_dataset,
    load_dataset_file,
)
from src.feature_engineering import feature_names
from src.model_training import TrainingRequest, train_classifier
from src.preprocessing import TRAINABLE_ATTACK_TYPES

CIC_HEADER: Final[tuple[str, ...]] = (
    "Label",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Average Packet Size",
    "Flow Packets/s",
)

CIC_NUMERIC_COLUMNS: Final[tuple[str, ...]] = CIC_HEADER[1:]

GENERIC_HEADER: Final[str] = (
    "Label,Flow Duration,Total Length,Total Packets,Avg Packet Size,"
    "Packets Per Second"
)

ALL_CIC_LABELS: Final[tuple[str, ...]] = (
    "BENIGN",
    "DoS Hulk",
    "DDoS",
    "PortScan",
    "Bot",
    "Brute Force",
    "Web Attack - XSS",
    "Infiltration",
)

SOURCE_TO_KWARG: Final[dict[str, str]] = {
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "fwd_packets",
    "Total Backward Packets": "bwd_packets",
    "Total Length of Fwd Packets": "fwd_bytes",
    "Total Length of Bwd Packets": "bwd_bytes",
    "Average Packet Size": "avg_packet_size",
    "Flow Packets/s": "packet_rate",
}

BASE_SOURCE_VALUES: Final[dict[str, float]] = {
    "Flow Duration": 10.0,
    "Total Fwd Packets": 2.0,
    "Total Backward Packets": 3.0,
    "Total Length of Fwd Packets": 100.0,
    "Total Length of Bwd Packets": 200.0,
    "Average Packet Size": 25.0,
    "Flow Packets/s": 4.0,
}


def _cic_row(
    label: object,
    *,
    flow_duration: object = 10,
    fwd_packets: object = 2,
    bwd_packets: object = 3,
    fwd_bytes: object = 100,
    bwd_bytes: object = 200,
    avg_packet_size: object = 25,
    packet_rate: object = 4,
) -> dict[str, object]:
    """Return one CIC row with distinct derived-source values."""
    return {
        "Label": label,
        "Flow Duration": flow_duration,
        "Total Fwd Packets": fwd_packets,
        "Total Backward Packets": bwd_packets,
        "Total Length of Fwd Packets": fwd_bytes,
        "Total Length of Bwd Packets": bwd_bytes,
        "Average Packet Size": avg_packet_size,
        "Flow Packets/s": packet_rate,
    }


def _expected_features(values: Mapping[str, float]) -> tuple[float, ...]:
    """Return the canonical vector for a set of CIC source measurements."""
    return (
        values["Flow Duration"],
        values["Total Length of Fwd Packets"]
        + values["Total Length of Bwd Packets"],
        values["Total Fwd Packets"] + values["Total Backward Packets"],
        values["Average Packet Size"],
        values["Flow Packets/s"],
    )


def _write_cic_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write CIC-shaped rows to a CSV file in pytest's temporary directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CIC_HEADER)
        for row in rows:
            writer.writerow([row[name] for name in CIC_HEADER])
    return path


def _write_large_cic_csv(path: Path, *, data_rows: int, excluded_at: int) -> Path:
    """Write a CIC CSV larger than the backend parser cap without buffering it."""
    benign = [_cic_row("BENIGN")[name] for name in CIC_HEADER]
    heartbleed = [_cic_row("Heartbleed")[name] for name in CIC_HEADER]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CIC_HEADER)
        for index in range(data_rows):
            writer.writerow(heartbleed if index == excluded_at else benign)
    return path


def _write_generic_csv(path: Path, rows: list[str]) -> Path:
    """Write generic canonical-header rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([GENERIC_HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def _paths(root: Path) -> MLPaths:
    """Return ML directories rooted at a temporary directory."""
    return MLPaths(
        ml_dir=root / "ml",
        dataset_dir=root / "dataset",
        model_dir=root / "models",
        outputs_dir=root / "outputs",
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("  BENIGN  ", AttackType.BENIGN),
        ("dos hulk", AttackType.DOS),
        ("GoldenEye", AttackType.DOS),
        ("DoS Slowloris", AttackType.DOS),
        ("DOS_SLOWHTTPTEST", AttackType.DOS),
        ("DDoS", AttackType.DDOS),
        ("PortScan", AttackType.PORT_SCAN),
        ("port scan", AttackType.PORT_SCAN),
        ("Bot", AttackType.BOT),
        ("Botnet-Ares", AttackType.BOT),
        ("Brute Force", AttackType.BRUTE_FORCE),
        ("FTP-Patator", AttackType.BRUTE_FORCE),
        ("SSH-Patator", AttackType.BRUTE_FORCE),
        ("Web Attack - Brute Force", AttackType.BRUTE_FORCE),
        ("Web Attack", AttackType.WEB_ATTACK),
        ("web attack-xss", AttackType.WEB_ATTACK),
        ("Web Attack \u2013 XSS", AttackType.WEB_ATTACK),
        ("Web Attack - SQL Injection", AttackType.WEB_ATTACK),
        ("SQL Injection", AttackType.WEB_ATTACK),
        ("XSS", AttackType.WEB_ATTACK),
        ("Infiltration", AttackType.INFILTRATION),
    ],
)
def test_label_normalization_handles_spelling_variants(
    label: object, expected: AttackType
) -> None:
    """Casing, whitespace, hyphens, dashes and underscores must not change class."""
    assert normalize_cic_label_to_trainable(label) is expected


@pytest.mark.parametrize("label", ["Heartbleed", "  HEARTBLEED ", "heart-bleed"])
def test_heartbleed_is_excluded_with_an_explicit_reason(label: str) -> None:
    """Heartbleed must not be mapped into the eight-class vocabulary."""
    from src.cic_ids2017_adapter import resolve_cic_label

    resolution = resolve_cic_label(label)
    assert resolution.attack_type is None
    assert resolution.code is CicExclusionReason.HEARTBLEED
    assert "eight-class Sentinel contract" in (resolution.reason or "")


@pytest.mark.parametrize("label", ["totally-new-class", "Unknown", ""])
def test_unknown_and_missing_labels_have_explicit_reasons(label: str) -> None:
    """Unrecognised and blank labels must be excluded with distinct reasons."""
    from src.cic_ids2017_adapter import resolve_cic_label

    resolution = resolve_cic_label(label)
    assert resolution.attack_type is None
    expected = (
        CicExclusionReason.MISSING_LABEL
        if not label.strip()
        else CicExclusionReason.UNKNOWN_LABEL
    )
    assert resolution.code is expected
    assert resolution.reason


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("Flow Duration", None, CicExclusionReason.MISSING_VALUE),
        ("Flow Duration", "", CicExclusionReason.MISSING_VALUE),
        ("Flow Duration", "not-a-number", CicExclusionReason.UNPARSABLE_VALUE),
        ("Flow Duration", True, CicExclusionReason.UNPARSABLE_VALUE),
        ("Flow Duration", "nan", CicExclusionReason.NON_FINITE_VALUE),
        ("Flow Packets/s", "inf", CicExclusionReason.NON_FINITE_VALUE),
    ],
)
def test_invalid_numeric_values_are_excluded_with_reasons(
    field: str, value: object, code: CicExclusionReason
) -> None:
    """Missing, unparsable, boolean and non-finite cells must be identifiable."""
    row = _cic_row("BENIGN")
    row[field] = value
    with pytest.raises(CicRowExcludedError) as excinfo:
        cic_row_to_example(row, 3)
    assert excinfo.value.code is code
    assert code.value in str(excinfo.value)


@pytest.mark.parametrize("column", CIC_NUMERIC_COLUMNS)
def test_underscore_numeric_strings_are_rejected_in_every_source_column(
    column: str,
) -> None:
    """A Python separator is not a backend numeric spelling, in any of the 7 columns."""
    row = _cic_row("BENIGN")
    row[column] = "1_0"
    with pytest.raises(CicRowExcludedError) as excinfo:
        cic_row_to_example(row, 0)
    assert excinfo.value.code is CicExclusionReason.UNPARSABLE_VALUE
    assert "1_0" in excinfo.value.reason


@pytest.mark.parametrize("column", CIC_NUMERIC_COLUMNS)
@pytest.mark.parametrize(("text", "number"), [("2.5", 2.5), ("1e1", 10.0)])
def test_decimal_and_scientific_notation_are_accepted_in_every_source_column(
    column: str, text: str, number: float
) -> None:
    """Backend-accepted decimal and scientific spellings must survive parsing."""
    row = _cic_row("BENIGN", **{SOURCE_TO_KWARG[column]: text})
    expected_values = dict(BASE_SOURCE_VALUES)
    expected_values[column] = number

    features, label = cic_row_to_example(row, 0)

    assert label is AttackType.BENIGN
    assert features == _expected_features(expected_values)


def test_non_finite_derived_value_is_excluded() -> None:
    """Finite source bytes that overflow when summed must be excluded."""
    row = _cic_row("BENIGN", fwd_bytes="1e308", bwd_bytes="1e308")
    with pytest.raises(CicRowExcludedError) as excinfo:
        cic_row_to_example(row, 7)
    assert excinfo.value.code is CicExclusionReason.NON_FINITE_DERIVED_VALUE
    assert "derived" in excinfo.value.reason


def test_conversion_records_exclusions_and_keeps_usable_rows() -> None:
    """A bad CIC row must not discard the whole file, but must be reported."""
    from src.cic_ids2017_adapter import convert_cic_rows

    rows = [
        _cic_row("DoS Hulk"),
        _cic_row("Heartbleed"),
        _cic_row("BENIGN", fwd_packets="oops"),
        _cic_row("Bot"),
    ]
    result = convert_cic_rows(rows)

    assert [example.label for example in result.examples] == [
        AttackType.DOS,
        AttackType.BOT,
    ]
    assert [(item.row_index, item.code) for item in result.exclusions] == [
        (1, CicExclusionReason.HEARTBLEED),
        (2, CicExclusionReason.UNPARSABLE_VALUE),
    ]


def test_feature_contract_is_exactly_five_production_features() -> None:
    """The adapter must derive the existing five features, in production order."""
    assert tuple(feature_names()) == CIC_EXPECTED_FEATURES
    assert len(feature_names()) == 5
    assert set(CIC_FEATURE_SOURCES) - {"label"} == set(feature_names())
    assert_cic_contract()


def test_feature_derivation_uses_forward_and_backward_columns() -> None:
    """Derived byte and packet totals must be summed and ordered correctly."""
    features, label = cic_row_to_example(_cic_row("DoS Hulk"), 0)
    assert label is AttackType.DOS
    assert features == (10.0, 300.0, 5.0, 25.0, 4.0)


def test_label_map_covers_exactly_the_eight_trainable_classes() -> None:
    """Every trainable Sentinel class must be reachable from a CIC label."""
    assert len(TRAINABLE_ATTACK_TYPES) == 8
    assert set(CIC_LABEL_TO_TRAINABLE.values()) == set(TRAINABLE_ATTACK_TYPES)
    assert_cic_contract()


def test_single_cic_file_may_have_partial_class_coverage(tmp_path: Path) -> None:
    """A capture day is not required to hold all eight trainable classes."""
    labels = ALL_CIC_LABELS[:-1]
    path = _write_cic_csv(tmp_path / "partial.csv", [_cic_row(label) for label in labels])

    loaded = load_dataset_file(path, dataset_format=DatasetFormat.CIC_IDS2017)

    assert set(loaded.dataset.labels) == {
        normalize_cic_label_to_trainable(label) for label in labels
    }
    assert loaded.rows_seen == len(labels)
    assert loaded.dataset.feature_order == feature_names()


def test_combined_cic_files_satisfy_eight_class_contract(tmp_path: Path) -> None:
    """Two incomplete capture files may together carry all eight classes."""
    paths = _paths(tmp_path)
    _write_cic_csv(
        paths.dataset_dir / "day1.csv",
        [_cic_row(label) for label in ALL_CIC_LABELS[:4]],
    )
    _write_cic_csv(
        paths.dataset_dir / "day2.csv",
        [_cic_row(label) for label in ALL_CIC_LABELS[4:]],
    )

    loaded = load_dataset(paths, dataset_format=DatasetFormat.CIC_IDS2017)

    assert set(loaded.dataset.labels) == set(TRAINABLE_ATTACK_TYPES)
    assert len(loaded.dataset) == 8
    assert loaded.rows_seen == 8


def test_explicit_file_selection_combines_only_selected_files(tmp_path: Path) -> None:
    """A caller may name the captures that form one logical dataset."""
    paths = _paths(tmp_path)
    first = _write_cic_csv(
        paths.dataset_dir / "day1.csv",
        [_cic_row(label) for label in ALL_CIC_LABELS[:4]],
    )
    second = _write_cic_csv(
        paths.dataset_dir / "day2.csv",
        [_cic_row(label) for label in ALL_CIC_LABELS[4:]],
    )
    _write_cic_csv(paths.dataset_dir / "excluded.csv", [_cic_row("Heartbleed")])

    loaded = load_dataset(
        paths, dataset_format=DatasetFormat.CIC_IDS2017, files=[first, second]
    )

    assert set(loaded.dataset.labels) == set(TRAINABLE_ATTACK_TYPES)
    assert loaded.rows_seen == 8
    assert loaded.exclusion_counts == {}


def test_combined_cic_files_missing_a_class_are_refused(tmp_path: Path) -> None:
    """The exact-eight check applies to the combined dataset, not to one file."""
    paths = _paths(tmp_path)
    _write_cic_csv(
        paths.dataset_dir / "day1.csv",
        [_cic_row(label) for label in ALL_CIC_LABELS[:4]],
    )
    _write_cic_csv(
        paths.dataset_dir / "day2.csv",
        [_cic_row(label) for label in ALL_CIC_LABELS[4:-1]],
    )

    with pytest.raises(CicClassCoverageError) as excinfo:
        load_dataset(paths, dataset_format=DatasetFormat.CIC_IDS2017)

    assert excinfo.value.missing == (AttackType.INFILTRATION,)
    assert "Infiltration" in str(excinfo.value)


def test_cic_reader_rejects_non_csv_files(tmp_path: Path) -> None:
    """A non-CSV file must be refused rather than skipped or misparsed."""
    path = _write_generic_csv(tmp_path / "day.json", ["BENIGN,10,1000,5,200,1"])

    with pytest.raises(DatasetLoadError, match="not a CSV file"):
        load_dataset_file(path, dataset_format=DatasetFormat.CIC_IDS2017)


def test_cic_directory_load_refuses_to_skip_non_csv_files(tmp_path: Path) -> None:
    """A mixed dataset directory must fail loudly, not drop a source file."""
    paths = _paths(tmp_path)
    _write_cic_csv(
        paths.dataset_dir / "day1.csv",
        [_cic_row(label) for label in ALL_CIC_LABELS],
    )
    (paths.dataset_dir / "extra.json").write_text("[]", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="will not silently skip non-CSV"):
        load_dataset(paths, dataset_format=DatasetFormat.CIC_IDS2017)


def test_large_cic_csv_beyond_backend_cap_loads_every_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CIC path must not inherit the generic parser's 100,000-row cap."""
    monkeypatch.setattr(log_parser, "MAX_LOG_ROWS", 2)
    path = _write_large_cic_csv(
        tmp_path / "huge.csv", data_rows=100_001, excluded_at=50_000
    )

    loaded = load_dataset_file(path, dataset_format=DatasetFormat.CIC_IDS2017)

    assert loaded.rows_seen == 100_001
    assert len(loaded.dataset) == 100_000
    assert loaded.exclusion_counts == {CicExclusionReason.HEARTBLEED: 1}
    assert [(item.row_index, item.code) for item in loaded.exclusions] == [
        (50_000, CicExclusionReason.HEARTBLEED)
    ]
    assert loaded.rows_seen == len(loaded.dataset) + sum(loaded.exclusion_counts.values())
    assert loaded.dataset.completeness.complete is True
    assert loaded.truncated_paths == ()


def test_aggregated_exclusion_counts_keep_row_level_reasons(tmp_path: Path) -> None:
    """Counts and row-level reasons must both survive a multi-file load."""
    paths = _paths(tmp_path)
    first_rows = [_cic_row(label) for label in ALL_CIC_LABELS[:4]]
    first_rows.append(_cic_row("Heartbleed"))
    second_rows = [_cic_row(label) for label in ALL_CIC_LABELS[4:]]
    second_rows.append(_cic_row("BENIGN", flow_duration="1_0"))
    second_rows.append(_cic_row("Bot"))
    _write_cic_csv(paths.dataset_dir / "day1.csv", first_rows)
    _write_cic_csv(paths.dataset_dir / "day2.csv", second_rows)

    loaded = load_dataset(paths, dataset_format=DatasetFormat.CIC_IDS2017)

    assert loaded.rows_seen == len(first_rows) + len(second_rows)
    assert loaded.exclusion_counts == {
        CicExclusionReason.HEARTBLEED: 1,
        CicExclusionReason.UNPARSABLE_VALUE: 1,
    }
    assert [(item.row_index, item.code) for item in loaded.exclusions] == [
        (4, CicExclusionReason.HEARTBLEED),
        (4, CicExclusionReason.UNPARSABLE_VALUE),
    ]
    assert loaded.rows_seen == len(loaded.dataset) + sum(loaded.exclusion_counts.values())


def test_underscore_value_from_csv_is_excluded_not_coerced(tmp_path: Path) -> None:
    """A CSV cell of ``1_0`` must not become ten in the training vector."""
    rows = [_cic_row(label) for label in ALL_CIC_LABELS]
    rows[0]["Flow Duration"] = "1_0"
    path = _write_cic_csv(tmp_path / "underscore.csv", rows)

    loaded = load_dataset_file(path, dataset_format=DatasetFormat.CIC_IDS2017)

    assert loaded.exclusion_counts == {CicExclusionReason.UNPARSABLE_VALUE: 1}
    assert loaded.exclusions[0].row_index == 0
    assert len(loaded.dataset) == 7


def test_cic_loader_uses_adapter_and_preserves_exclusions(tmp_path: Path) -> None:
    """The CIC reader must derive the five features and report dropped rows."""
    rows = [_cic_row(label) for label in ALL_CIC_LABELS]
    rows.append(_cic_row("Heartbleed"))
    path = _write_cic_csv(tmp_path / "cic.csv", rows)

    loaded = load_dataset_file(path, dataset_format=DatasetFormat.CIC_IDS2017)

    assert loaded.dataset_format is DatasetFormat.CIC_IDS2017
    assert loaded.mapped_columns == CIC_FEATURE_SOURCES
    assert len(loaded.dataset) == 8
    assert set(loaded.dataset.class_counts) == set(TRAINABLE_ATTACK_TYPES)
    assert loaded.dataset.examples[0].features == (10.0, 300.0, 5.0, 25.0, 4.0)
    assert [(item.row_index, item.code) for item in loaded.exclusions] == [
        (8, CicExclusionReason.HEARTBLEED)
    ]
    assert loaded.exclusion_counts == {CicExclusionReason.HEARTBLEED: 1}
    assert loaded.rows_seen == 9


def test_generic_loader_remains_the_default(tmp_path: Path) -> None:
    """Existing production inference mapping must be unchanged by the adapter."""
    path = _write_generic_csv(
        tmp_path / "generic.csv",
        ["BENIGN,10,1000,5,200,1", "DoS Hulk,20,2000,10,200,2"],
    )

    loaded = load_dataset_file(path)

    assert loaded.dataset_format is DatasetFormat.GENERIC
    assert loaded.mapped_columns["total_bytes"] == "Total Length"
    assert loaded.dataset.examples[0].features == (10.0, 1000.0, 5.0, 200.0, 1.0)
    assert loaded.exclusions == ()
    assert loaded.dataset.completeness.complete is True


def test_unsupported_dataset_format_is_refused(tmp_path: Path) -> None:
    """An unknown reader name must fail before any file is read."""
    path = _write_generic_csv(tmp_path / "generic.csv", ["BENIGN,10,1000,5,200,1"])
    with pytest.raises(UnsupportedDatasetFormatError):
        load_dataset_file(path, dataset_format="not-a-reader")


def test_single_file_truncation_is_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capped generic parse must be visible on the dataset contract."""
    monkeypatch.setattr(log_parser, "MAX_LOG_ROWS", 2)
    path = _write_generic_csv(
        tmp_path / "large.csv",
        [
            "BENIGN,10,1000,5,200,1",
            "DoS,20,2000,10,200,2",
            "Bot,30,3000,15,200,3",
        ],
    )

    loaded = load_dataset_file(path)

    assert loaded.truncated is True
    assert loaded.truncated_paths == (path,)
    assert len(loaded.dataset) == 2
    assert loaded.dataset.completeness.complete is False
    assert loaded.dataset.completeness.truncated_paths == (path,)
    assert "truncated" in loaded.dataset.completeness.reasons[0]


def test_combined_load_reports_every_truncated_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A combined dataset must not look complete when any source was capped."""
    monkeypatch.setattr(log_parser, "MAX_LOG_ROWS", 2)
    paths = _paths(tmp_path)
    first = _write_generic_csv(
        paths.dataset_dir / "day1.csv",
        [
            "BENIGN,10,1000,5,200,1",
            "DoS,20,2000,10,200,2",
            "Bot,30,3000,15,200,3",
        ],
    )
    second = _write_generic_csv(
        paths.dataset_dir / "day2.csv",
        ["BENIGN,40,4000,20,200,4", "DoS,50,5000,25,200,5"],
    )

    loaded = load_dataset(paths)

    assert loaded.truncated is True
    assert loaded.truncated_paths == (first,)
    assert second not in loaded.truncated_paths
    assert loaded.dataset.completeness.complete is False
    assert loaded.dataset.completeness.truncated_paths == (first,)


def test_training_refuses_an_incomplete_load_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated dataset must be refused before an estimator is considered."""
    from src.dataset_validation import IncompleteDatasetError

    monkeypatch.setattr(log_parser, "MAX_LOG_ROWS", 2)
    path = _write_generic_csv(
        tmp_path / "large.csv",
        ["BENIGN,10,1000,5,200,1", "DoS,20,2000,10,200,2", "Bot,30,3,15,2,3"],
    )
    dataset = load_dataset_file(path).dataset
    request = TrainingRequest(dataset=dataset, model_name=DEFAULT_MODEL)

    with pytest.raises(IncompleteDatasetError):
        request.validate()
    with pytest.raises(IncompleteDatasetError):
        train_classifier(request)


def test_disk_backed_cic_rows_and_exclusions(tmp_path: Path) -> None:
    """Disk storage must keep both the compact rows and row-level reasons."""
    rows = [_cic_row(label) for label in ALL_CIC_LABELS]
    rows.append(_cic_row("Heartbleed"))
    path = _write_cic_csv(tmp_path / "cic.csv", rows)
    storage_dir = tmp_path / "storage"
    exclusion_path = tmp_path / "exclusions.jsonl"

    loaded = load_dataset_file(
        path,
        dataset_format=DatasetFormat.CIC_IDS2017,
        storage=DatasetStorage.DISK,
        storage_dir=storage_dir,
        exclusions_path=exclusion_path,
    )

    assert loaded.dataset.storage is DatasetStorage.DISK
    assert len(loaded.dataset.storage_paths) == 2
    assert all(item.exists() for item in loaded.dataset.storage_paths)
    assert exclusion_path.exists()
    assert loaded.dataset.examples[0].features == (10.0, 300.0, 5.0, 25.0, 4.0)
    assert [(item.row_index, item.code) for item in loaded.exclusions] == [
        (8, CicExclusionReason.HEARTBLEED)
    ]
    assert loaded.exclusion_counts == {CicExclusionReason.HEARTBLEED: 1}


def test_disk_backed_combined_load_keeps_every_reason(tmp_path: Path) -> None:
    """A combined disk-backed load must not collapse per-file exclusions."""
    paths = _paths(tmp_path)
    first_rows = [_cic_row(label) for label in ALL_CIC_LABELS[:4]]
    first_rows.append(_cic_row("Heartbleed"))
    second_rows = [_cic_row(label) for label in ALL_CIC_LABELS[4:]]
    second_rows.append(_cic_row("BENIGN", flow_duration="1_0"))
    _write_cic_csv(paths.dataset_dir / "day1.csv", first_rows)
    _write_cic_csv(paths.dataset_dir / "day2.csv", second_rows)

    loaded = load_dataset(
        paths,
        dataset_format=DatasetFormat.CIC_IDS2017,
        storage=DatasetStorage.DISK,
        storage_dir=tmp_path / "storage",
        exclusion_dir=tmp_path / "exclusions",
    )

    assert loaded.dataset.storage is DatasetStorage.DISK
    assert len(loaded.dataset) == 8
    assert [(item.row_index, item.code) for item in loaded.exclusions] == [
        (4, CicExclusionReason.HEARTBLEED),
        (4, CicExclusionReason.UNPARSABLE_VALUE),
    ]
    assert loaded.exclusion_counts == {
        CicExclusionReason.HEARTBLEED: 1,
        CicExclusionReason.UNPARSABLE_VALUE: 1,
    }
    assert loaded.rows_seen == len(first_rows) + len(second_rows)


def test_disk_backed_load_with_no_usable_rows_fails_clearly(tmp_path: Path) -> None:
    """An all-excluded disk-backed file must not be confused with an empty mmap."""
    path = _write_cic_csv(
        tmp_path / "excluded.csv", [_cic_row("Heartbleed")]
    )

    with pytest.raises(DatasetLoadError, match="no usable CIC rows"):
        load_dataset_file(
            path,
            dataset_format=DatasetFormat.CIC_IDS2017,
            storage=DatasetStorage.DISK,
            storage_dir=tmp_path / "storage",
            exclusions_path=tmp_path / "exclusions.jsonl",
        )


def test_disk_backed_failure_cleans_all_storage_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed single-file load must leave no rows or exclusions behind."""
    rows = [_cic_row(label) for label in ALL_CIC_LABELS]
    path = _write_cic_csv(tmp_path / "cic.csv", rows)
    storage_dir = tmp_path / "storage"
    exclusion_path = tmp_path / "exclusions.jsonl"

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("validation failed")

    monkeypatch.setattr(cic_adapter, "validate_arrays", fail)

    with pytest.raises(RuntimeError, match="validation failed"):
        load_dataset_file(
            path,
            dataset_format=DatasetFormat.CIC_IDS2017,
            storage=DatasetStorage.DISK,
            storage_dir=storage_dir,
            exclusions_path=exclusion_path,
        )

    assert not storage_dir.exists() or not any(storage_dir.iterdir())
    assert not exclusion_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_combined_disk_failure_cleans_intermediate_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed combined load must remove per-file and combined artifacts."""
    paths = _paths(tmp_path)
    _write_cic_csv(
        paths.dataset_dir / "day1.csv",
        [_cic_row(label) for label in ALL_CIC_LABELS[:4]],
    )
    _write_cic_csv(
        paths.dataset_dir / "day2.csv",
        [_cic_row(label) for label in ALL_CIC_LABELS[4:]],
    )
    storage_dir = tmp_path / "storage"
    exclusion_dir = tmp_path / "exclusions"

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("coverage failed")

    monkeypatch.setattr(data_loader, "validate_cic_examples", fail)

    with pytest.raises(RuntimeError, match="coverage failed"):
        load_dataset(
            paths,
            dataset_format=DatasetFormat.CIC_IDS2017,
            storage=DatasetStorage.DISK,
            storage_dir=storage_dir,
            exclusion_dir=exclusion_dir,
        )

    assert not storage_dir.exists() or not any(storage_dir.iterdir())
    assert not exclusion_dir.exists() or not any(exclusion_dir.iterdir())
    assert not list(tmp_path.glob(".*.tmp"))
