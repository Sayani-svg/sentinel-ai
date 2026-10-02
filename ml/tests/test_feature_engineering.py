"""Feature vectors must be positional, complete, and derived from one source of truth.

The ordering tests here are the important ones. A feature vector is a tuple of
floats with no names attached, so a reordering anywhere between the enum and the
estimator produces predictions that are confidently wrong rather than obviously
broken. Asserting the exact order, and asserting positions against distinct
values, is what catches that.
"""

from __future__ import annotations

import pytest
from app.services.prediction_service import DetectionFeature

from src.feature_engineering import (
    NUMERIC_FEATURES,
    FeatureExtractionError,
    FeatureOrderError,
    MissingFeatureError,
    assert_feature_order,
    feature_coverage,
    feature_names,
    missing_features,
    to_feature_matrix,
    to_feature_space,
    to_feature_vector,
)


def test_label_is_excluded_from_the_feature_vector() -> None:
    """The label column is the target, not a feature."""
    assert DetectionFeature.LABEL not in NUMERIC_FEATURES
    assert DetectionFeature.LABEL not in feature_names()


def test_features_are_the_numeric_measurements_in_declaration_order() -> None:
    """The vector must carry every non-label feature, in enum order."""
    expected = tuple(
        str(feature)
        for feature in DetectionFeature
        if feature is not DetectionFeature.LABEL
    )
    assert feature_names() == expected
    assert NUMERIC_FEATURES == tuple(
        feature
        for feature in DetectionFeature
        if feature is not DetectionFeature.LABEL
    )


def test_vector_positions_match_declaration_order(make_record) -> None:
    """Each position must hold the value of the feature named for it.

    Distinct values per feature, so a transposition cannot pass.
    """
    ordered = {name: float(index + 1) for index, name in enumerate(feature_names())}
    record = make_record(**ordered)
    assert to_feature_vector(record) == (1.0, 2.0, 3.0, 4.0, 5.0)


def test_vector_round_trips_a_complete_record(make_record) -> None:
    """A complete record must yield one value per declared feature."""
    vector = to_feature_vector(make_record())
    assert len(vector) == len(feature_names())
    assert all(isinstance(value, float) for value in vector)


def test_absent_feature_is_refused_not_imputed(make_record) -> None:
    """A record missing a feature must be refused, naming the feature."""
    record = make_record(drop=(str(DetectionFeature.PACKET_RATE),))
    assert missing_features(record) == (DetectionFeature.PACKET_RATE,)

    with pytest.raises(MissingFeatureError) as excinfo:
        to_feature_vector(record)

    error = excinfo.value
    assert error.missing == (DetectionFeature.PACKET_RATE,)
    assert DetectionFeature.PACKET_RATE in str(error)


def test_refusal_reports_the_offending_row(make_record) -> None:
    """The row index must reach the caller so the source line can be found."""
    record = make_record(drop=(DetectionFeature.TOTAL_BYTES,))
    with pytest.raises(MissingFeatureError) as excinfo:
        to_feature_vector(record, row_index=17)
    assert excinfo.value.row_index == 17


def test_every_missing_feature_is_named_at_once(make_record) -> None:
    """All absent features must be reported together, not one per attempt."""
    record = make_record(drop=(DetectionFeature.TOTAL_BYTES, DetectionFeature.TOTAL_PACKETS))
    with pytest.raises(MissingFeatureError) as excinfo:
        to_feature_vector(record)
    assert excinfo.value.missing == (DetectionFeature.TOTAL_BYTES, DetectionFeature.TOTAL_PACKETS)


def test_missing_feature_error_is_a_feature_extraction_error(make_record) -> None:
    """Callers should be able to catch the family, not just this case."""
    with pytest.raises(FeatureExtractionError):
        to_feature_vector(make_record(drop=(DetectionFeature.TOTAL_BYTES,)))


def test_matrix_rows_follow_record_order(make_record) -> None:
    """The matrix must preserve record order, one row per record."""
    name = str(DetectionFeature.TOTAL_BYTES)
    first = make_record(**{name: 1.0})
    second = make_record(**{name: 2.0})
    rows = to_feature_matrix([first, second])
    assert len(rows) == 2
    position = feature_names().index(name)
    assert rows[0][position] == 1.0
    assert rows[1][position] == 2.0


def test_matrix_reports_the_row_it_refused(make_record) -> None:
    """A bad record deep in a batch must be identified by position."""
    records = [make_record(), make_record(drop=(DetectionFeature.TOTAL_BYTES,))]
    with pytest.raises(MissingFeatureError) as excinfo:
        to_feature_matrix(records)
    assert excinfo.value.row_index == 1


def test_complete_records_report_no_gaps(make_record) -> None:
    """A complete record must report nothing missing."""
    assert missing_features(make_record()) == ()


def test_coverage_counts_rows_carrying_each_feature(make_record) -> None:
    """Coverage must expose a column present in the file but absent from rows."""
    records = [
        make_record(),
        make_record(drop=(DetectionFeature.PACKET_RATE,)),
        make_record(drop=(DetectionFeature.PACKET_RATE, DetectionFeature.AVG_PACKET_SIZE)),
    ]
    coverage = feature_coverage(records)
    assert set(coverage) == set(feature_names())
    assert coverage[DetectionFeature.PACKET_RATE] == 1
    assert coverage[DetectionFeature.AVG_PACKET_SIZE] == 2
    assert coverage[DetectionFeature.TOTAL_BYTES] == 3


def test_coverage_is_reported_in_vector_order(make_record) -> None:
    """Coverage must follow vector order so it lines up with a matrix."""
    assert list(feature_coverage([make_record()])) == list(feature_names())


def test_matching_feature_order_is_accepted() -> None:
    """This build's own order must pass."""
    assert_feature_order(feature_names())


def test_reordered_features_are_refused() -> None:
    """The same features in a different order must be refused.

    This is the failure a positional vector cannot report on its own: every value
    is a valid float and every prediction looks reasonable.
    """
    reversed_order = list(reversed(feature_names()))
    with pytest.raises(FeatureOrderError) as excinfo:
        assert_feature_order(reversed_order)
    assert "different order" in str(excinfo.value)


def test_unknown_feature_order_is_refused() -> None:
    """An order naming features this build lacks must be refused."""
    with pytest.raises(FeatureOrderError):
        assert_feature_order([*feature_names(), "SOMETHING_ELSE"])


def test_empty_feature_order_is_refused() -> None:
    """An empty order must not pass as a degenerate match."""
    with pytest.raises(FeatureOrderError):
        assert_feature_order([])


def test_to_feature_space_uses_the_application_pipeline(make_record) -> None:
    """A parsed log must arrive in the same space the detector uses."""
    from app.services.log_parser import parse_json_log

    parsed = parse_json_log(
        '[{"Label": "DoS", "Flow Duration": 12, "Total Length": 300, '
        '"Total Packets": 4, "Avg Packet Size": 75, "Packets Per Second": 2}]'
    )
    detection_input = to_feature_space(parsed)
    assert detection_input.row_count == 1
    assert detection_input.has_label
    record = detection_input.records[0]
    assert to_feature_vector(record) == (12.0, 300.0, 4.0, 75.0, 2.0)