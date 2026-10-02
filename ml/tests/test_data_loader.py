"""The loader must read a benchmark through the application's own parser.

The files written here are test inputs written to pytest's ``tmp_path`` so the
reader can be exercised against real bytes: a real header line, real delimiter
sniffing, real type coercion. They are not a dataset. Nothing in ``ml/dataset``
is created, and no row here is treated as representative of real traffic -- the
rows exist only to check that a header maps to the right canonical feature and a
label to the right class.

The refusals matter more than the successes. ``ml/dataset`` is empty in this
checkout, so the paths that must be trustworthy are the ones that stop a
half-compatible file from producing a trained model.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from src.config import MLPaths
from src.data_loader import (
    SUPPORTED_SUFFIXES,
    DatasetLoadError,
    NoUsableFeaturesError,
    UnlabelledDatasetError,
    build_examples,
    discover_dataset_files,
    load_dataset,
    load_dataset_file,
)
from src.feature_engineering import feature_names, to_feature_space
from src.preprocessing import require_trainable_label

#: Headers whose normalised forms resolve to every declared feature.
FULL_HEADER = (
    "Label,Flow Duration,Total Length,Total Packets,Avg Packet Size,"
    "Packets Per Second"
)


def _paths(root: Path) -> MLPaths:
    """Return an :class:`MLPaths` rooted at a temporary directory.

    Args:
        root: The temporary root to use.

    Returns:
        MLPaths: Directories under ``root``.
    """
    return MLPaths(
        ml_dir=root / "ml",
        dataset_dir=root / "dataset",
        model_dir=root / "models",
        outputs_dir=root / "outputs",
    )


def _write_csv(path: Path, rows: list[str], header: str = FULL_HEADER) -> Path:
    """Write a CSV of the given header and data rows.

    Args:
        path: The file to write.
        rows: Data lines, without a header.
        header: The header line.

    Returns:
        Path: The written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_supported_suffixes_match_implemented_readers() -> None:
    """The loader must only advertise formats the application can read."""
    from app.schemas.upload import LogFileFormat

    implemented = {f".{member.value}" for member in LogFileFormat}
    assert set(SUPPORTED_SUFFIXES) == implemented


def test_discovery_finds_data_files_in_name_order(tmp_path: Path) -> None:
    """Discovery must find data files, sorted, ignoring hidden and other files."""
    paths = _paths(tmp_path)
    paths.dataset_dir.mkdir(parents=True)
    _write_csv(paths.dataset_dir / "b.csv", ["a,b"])
    _write_csv(paths.dataset_dir / "a.csv", ["a,b"])
    _write_csv(paths.dataset_dir / "notes.txt", ["a,b"])
    (paths.dataset_dir / ".hidden.csv").write_text("a,b", encoding="utf-8")

    assert [p.name for p in discover_dataset_files(paths)] == ["a.csv", "b.csv"]


def test_discovery_on_missing_directory_is_empty(tmp_path: Path) -> None:
    """A checkout with no dataset directory must report nothing, not raise."""
    assert discover_dataset_files(_paths(tmp_path)) == ()


def test_load_reads_features_and_labels(tmp_path: Path) -> None:
    """A fully mapped file must yield one row per record with its label."""
    path = _write_csv(
        tmp_path / "day.csv",
        ["BENIGN,10,1000,5,200,1", "DoS Hulk,20,2000,10,200,2"],
    )

    loaded = load_dataset_file(path)

    assert len(loaded.dataset) == 2
    assert set(loaded.dataset.class_counts) == {"Benign", "DoS"}
    first = loaded.dataset.examples[0]
    assert first.features == (10.0, 1000.0, 5.0, 200.0, 1.0)


def test_load_records_its_column_mapping(tmp_path: Path) -> None:
    """The mapping from canonical names to source headers must be reported.

    This is what lets a reviewer confirm a benchmark's columns were understood
    rather than approximated.
    """
    path = _write_csv(
        tmp_path / "day.csv", ["BENIGN,10,1000,5,200,1", "DoS,20,2000,10,200,2"]
    )
    loaded = load_dataset_file(path)
    assert loaded.mapped_columns == {
        "label": "Label",
        "flow_duration": "Flow Duration",
        "total_bytes": "Total Length",
        "total_packets": "Total Packets",
        "avg_packet_size": "Avg Packet Size",
        "packet_rate": "Packets Per Second",
    }


def test_load_resolves_every_spelling_of_a_class(tmp_path: Path) -> None:
    """Benchmark spellings must collapse onto the classes the app defines."""
    path = _write_csv(
        tmp_path / "day.csv",
        [
            "BENIGN,10,1000,5,200,1",
            "DoS Hulk,20,2000,10,200,2",
            "DDoS,30,3000,15,200,3",
            "PortScan,40,4000,20,200,4",
            "Bot,50,5000,25,200,5",
            "BruteForce,60,6000,30,200,6",
            "Web Attack - XSS,70,7000,35,200,7",
            "Infiltration,80,8000,40,200,8",
        ],
    )

    loaded = load_dataset_file(path)

    assert {row.label.value for row in loaded.dataset.examples} == {
        "Benign",
        "DoS",
        "DDoS",
        "PortScan",
        "Bot",
        "BruteForce",
        "WebAttack",
        "Infiltration",
    }


def test_load_reads_json(tmp_path: Path) -> None:
    """JSON must load through the same path as CSV."""
    path = tmp_path / "day.json"
    path.write_text(
        '[{"Label": "BENIGN", "Flow Duration": 10, "Total Length": 1000, '
        '"Total Packets": 5, "Avg Packet Size": 200, "Packets Per Second": 1},'
        ' {"Label": "DoS", "Flow Duration": 20, "Total Length": 2000, '
        '"Total Packets": 10, "Avg Packet Size": 200, "Packets Per Second": 2}]',
        encoding="utf-8",
    )

    loaded = load_dataset_file(path)
    assert loaded.dataset.examples[0].label.value == "Benign"


def test_file_with_no_usable_features_is_refused(tmp_path: Path) -> None:
    """Headers this build does not understand must be refused, not coerced.

    Reading unknown columns as zero would train a model on the absence of data
    and report it as a measurement.
    """
    path = _write_csv(
        tmp_path / "odd.csv", ["1,2,3"], header="Alpha,Beta,Gamma"
    )

    with pytest.raises(NoUsableFeaturesError) as excinfo:
        load_dataset_file(path)

    error = excinfo.value
    assert error.path == path
    assert "Alpha" in str(error)
    assert set(error.mapped) == set()


def test_unlabelled_file_is_refused(tmp_path: Path) -> None:
    """A file stating no class cannot be learned from."""
    path = _write_csv(
        tmp_path / "day.csv",
        ["10,1000,5,200,1", "20,2000,10,200,2"],
        header="Flow Duration,Total Length,Total Packets,Avg Packet Size,Packets Per Second",
    )

    with pytest.raises(UnlabelledDatasetError) as excinfo:
        load_dataset_file(path)
    assert excinfo.value.path == path


def test_unrecognised_label_spelling_is_refused(tmp_path: Path) -> None:
    """A label naming no known class must be refused rather than guessed."""
    path = _write_csv(
        tmp_path / "day.csv",
        [
            "BENIGN,10,1000,5,200,1",
            "totally-new-class,20,2000,10,200,2",
        ],
    )

    from src.preprocessing import LabelMappingError

    with pytest.raises(LabelMappingError):
        load_dataset_file(path)


def test_single_class_file_is_refused(tmp_path: Path) -> None:
    """A file of one class must not reach training."""
    path = _write_csv(
        tmp_path / "day.csv", ["BENIGN,10,1000,5,200,1", "BENIGN,20,2000,10,200,2"]
    )

    from src.dataset_validation import InsufficientClassCoverageError

    with pytest.raises(InsufficientClassCoverageError):
        load_dataset_file(path)


def test_row_missing_a_feature_is_refused(tmp_path: Path) -> None:
    """A row with a blank feature must be refused, not imputed.

    ``Total Length`` blank means the flow's size was not measured, which is not
    the same statement as the flow having moved zero bytes.
    """
    path = _write_csv(
        tmp_path / "day.csv",
        ["BENIGN,10,1000,5,200,1", "BENIGN,20,,10,200,2"],
    )

    from src.feature_engineering import MissingFeatureError

    with pytest.raises(MissingFeatureError):
        load_dataset_file(path)


def test_load_dataset_combines_files(tmp_path: Path) -> None:
    """Several capture files must load into one dataset."""
    paths = _paths(tmp_path)
    _write_csv(paths.dataset_dir / "day1.csv", ["BENIGN,10,1000,5,200,1", "DoS,20,2000,10,200,2"])
    _write_csv(paths.dataset_dir / "day2.csv", ["BENIGN,30,3000,15,200,3", "DoS,40,4000,20,200,4"])

    loaded = load_dataset(paths)

    assert len(loaded.dataset) == 4
    assert loaded.dataset.class_counts == {"Benign": 2, "DoS": 2}


def test_load_dataset_names_the_directory_it_searched(tmp_path: Path) -> None:
    """With no dataset present, the error must say where it looked.

    The empty ``ml/dataset`` in a fresh checkout is the expected state, so this
    message is the one an operator will actually read.
    """
    paths = _paths(tmp_path)

    with pytest.raises(DatasetLoadError) as excinfo:
        load_dataset(paths)

    message = str(excinfo.value)
    assert str(paths.dataset_dir) in message
    assert "will not generate" in message or "nothing" in message.lower()


def test_unsupported_extension_is_refused(tmp_path: Path) -> None:
    """A file the application cannot read must be refused by name."""
    path = tmp_path / "day.txt"
    path.write_text("Label\nBENIGN\n", encoding="utf-8")

    with pytest.raises(DatasetLoadError):
        load_dataset_file(path)


def test_build_examples_requires_labels_by_default(tmp_path: Path) -> None:
    """Row building must refuse unlabelled records unless told not to."""
    from app.services.log_parser import parse_csv_log

    parsed = parse_csv_log(
        io.StringIO(
            "Flow Duration,Total Length,Total Packets,Avg Packet Size,Packets Per Second\n"
            "10,1000,5,200,1\n20,2000,10,200,2\n"
        )
    )
    detection_input = to_feature_space(parsed)

    with pytest.raises(UnlabelledDatasetError):
        build_examples(detection_input, path=Path("<test>"), columns=feature_names())

    assert build_examples(
        detection_input,
        path=Path("<test>"),
        columns=feature_names(),
        require_labels=False,
    ) == ()


def test_require_trainable_label_is_the_door_for_build_examples() -> None:
    """Row building must resolve labels through the shared vocabulary."""
    assert require_trainable_label("DoS Hulk").value == "DoS"
    assert require_trainable_label("BENIGN").value == "Benign"