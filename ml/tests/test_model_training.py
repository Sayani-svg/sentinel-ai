"""The classifier contract, and the refusal to train on nothing.

The rule-based wrapper is the behaviour the application has today, so these tests
pin two things: that wrapping it changed nothing about the verdicts, and that the
provenance stamp is now attached. The training tests pin the *refusals* -- a
training entry point that quietly did nothing would be indistinguishable, from a
run log, from one that produced a model.
"""

from __future__ import annotations

import io
from datetime import datetime

import pytest
from app.schemas.prediction import AttackType
from app.services.log_parser import parse_csv_log
from app.services.prediction_service import DetectionInput, detect_attack

from src.config import DEFAULT_MODEL, MODEL_VERSION
from src.dataset_validation import dataset_from_records
from src.feature_engineering import feature_names, to_feature_space
from src.model_training import (
    MODEL_CONFIDENCE_FLOOR,
    Classifier,
    ClassifierSource,
    ClassifiedResult,
    EstimatorUnavailableError,
    RuleBasedClassifier,
    TrainingDataUnavailableError,
    TrainingError,
    TrainingRequest,
    assert_decision_range,
    build_training_run,
    summarise_run,
    train_classifier,
)


def _detection_input(csv: str) -> DetectionInput:
    """Return a prepared detection input from CSV text.

    Args:
        csv: The CSV text, header included.

    Returns:
        DetectionInput: The records in the canonical feature space.
    """
    return to_feature_space(parse_csv_log(io.StringIO(csv)))


def _flooding_file() -> DetectionInput:
    """Return a file the rules classify as a flood.

    Returns:
        DetectionInput: A single very large, very fast flow.
    """
    return _detection_input(
        "Flow Duration,Total Length,Total Packets,Avg Packet Size,Packets Per Second\n"
        "100,5000000,4000,1250,4000\n"
    )


def _quiet_file() -> DetectionInput:
    """Return a file the rules classify as benign.

    Returns:
        DetectionInput: A single small, slow flow.
    """
    return _detection_input(
        "Flow Duration,Total Length,Total Packets,Avg Packet Size,Packets Per Second\n"
        "1000000,800,4,200,0.001\n"
    )


def _dataset():
    """Return a small validated dataset.

    Returns:
        TrainingDataset: Two rows, one per class.
    """
    width = len(feature_names())
    return dataset_from_records(
        [
            ((1.0,) * width, AttackType.BENIGN),
            ((2.0,) * width, AttackType.DOS),
        ]
    )


def test_rule_classifier_reports_the_rules_as_its_source() -> None:
    """The wrapper must declare itself the rule-based path."""
    assert RuleBasedClassifier().source is ClassifierSource.RULES


def test_rule_classifier_declares_the_canonical_feature_order() -> None:
    """It must read the same features, in the same order, as training."""
    assert RuleBasedClassifier().feature_order == feature_names()


def test_rule_classifier_preserves_the_existing_verdict() -> None:
    """Wrapping must not change any verdict the detector already produced.

    The point of the wrapper is to be swappable. If it altered a verdict, every
    existing detection would be a behaviour change rather than a refactor.
    """
    detection_input = _flooding_file()
    expected = detect_attack(detection_input)

    result = RuleBasedClassifier().classify(detection_input)

    assert result.attack_type is expected.attack_type
    assert result.confidence == expected.confidence


def test_rule_classifier_matches_the_detector_on_several_files() -> None:
    """Agreement must hold across files, not just one convenient case."""
    classifier = RuleBasedClassifier()
    for detection_input in (_flooding_file(), _quiet_file()):
        expected = detect_attack(detection_input)
        result = classifier.classify(detection_input)
        assert (result.attack_type, result.confidence) == (
            expected.attack_type,
            expected.confidence,
        )


def test_rule_results_name_the_rules_that_fired() -> None:
    """The verdict must carry its justification for the log."""
    detection_input = _flooding_file()
    result = RuleBasedClassifier().classify(detection_input)
    assert result.detail
    assert result.detail == "; ".join(detect_attack(detection_input).matched_rules)


def test_rule_results_are_stamped_as_rules() -> None:
    """A heuristic verdict must be distinguishable from a learned one."""
    result = RuleBasedClassifier().classify(_flooding_file())
    assert result.source is ClassifierSource.RULES


def test_classifier_is_an_abstract_contract() -> None:
    """The contract must not be instantiable on its own."""
    with pytest.raises(TypeError):
        Classifier()  # type: ignore[abstract]


def test_rule_classifier_is_a_classifier() -> None:
    """The existing detector must satisfy the shared contract."""
    assert isinstance(RuleBasedClassifier(), Classifier)


def test_training_request_defaults_match_application_settings() -> None:
    """Request defaults must mirror the settings the application already has."""
    from app.core.config import Settings

    request = TrainingRequest(dataset=_dataset(), model_name=DEFAULT_MODEL)
    settings = Settings()
    assert request.version == settings.MODEL_VERSION == MODEL_VERSION
    assert request.random_state == settings.RANDOM_STATE


def test_training_request_carries_the_dataset_feature_order() -> None:
    """A request must train against the order the dataset was validated with."""
    request = TrainingRequest(dataset=_dataset(), model_name=DEFAULT_MODEL)
    assert request.feature_order == _dataset().feature_order


def test_request_with_mismatched_feature_order_is_refused() -> None:
    """A request must not train against a different order than its dataset.

    The resulting artifact would be served against vectors it never saw, and
    positional features would hide that.
    """
    request = TrainingRequest(
        dataset=_dataset(),
        model_name=DEFAULT_MODEL,
        feature_order=tuple(reversed(feature_names())),
    )
    with pytest.raises(TrainingError):
        request.validate()


def test_request_is_validated_before_the_estimator_check() -> None:
    """A malformed request must fail on the data, not on the missing estimator.

    Otherwise a caller learns "no estimator wired up" for a dataset that was
    itself unusable, and fixes the wrong problem. Validating first means the
    error names whichever thing is actually broken.
    """
    request = TrainingRequest(
        dataset=_dataset(),
        model_name=DEFAULT_MODEL,
        feature_order=tuple(reversed(feature_names())),
    )

    with pytest.raises(TrainingError) as excinfo:
        train_classifier(request)

    assert not isinstance(excinfo.value, EstimatorUnavailableError)


def test_build_training_run_records_what_was_trained() -> None:
    """A run's metadata must state the artifact's positional contract."""
    request = TrainingRequest(dataset=_dataset(), model_name=DEFAULT_MODEL)
    run = build_training_run(request, trained_at=datetime(2026, 1, 1))

    assert run.model_name == DEFAULT_MODEL
    assert run.version == MODEL_VERSION
    assert run.feature_order == feature_names()
    assert run.rows_trained == 2
    assert run.classes == (AttackType.BENIGN, AttackType.DOS)
    assert run.trained_at == datetime(2026, 1, 1)


def test_run_classes_are_in_code_order_not_seen_order() -> None:
    """Class order must be the integer codes an estimator returns columns in."""
    width = len(feature_names())
    dataset = dataset_from_records(
        [
            ((1.0,) * width, AttackType.DOS),
            ((2.0,) * width, AttackType.BENIGN),
        ]
    )
    run = build_training_run(
        TrainingRequest(dataset=dataset, model_name=DEFAULT_MODEL),
        trained_at=datetime(2026, 1, 1),
    )
    assert run.classes == (AttackType.BENIGN, AttackType.DOS)


def test_run_records_class_counts() -> None:
    """Imbalance at training time must be recorded, not inferred later."""
    width = len(feature_names())
    dataset = dataset_from_records(
        [
            ((1.0,) * width, AttackType.BENIGN),
            ((1.0,) * width, AttackType.BENIGN),
            ((2.0,) * width, AttackType.DOS),
        ]
    )
    run = build_training_run(
        TrainingRequest(dataset=dataset, model_name=DEFAULT_MODEL),
        trained_at=datetime(2026, 1, 1),
    )
    assert run.class_counts == {AttackType.BENIGN: 2, AttackType.DOS: 1}


def test_training_is_refused_without_an_estimator() -> None:
    """Training must raise rather than silently do nothing.

    A no-op returning ``None`` would let a caller proceed as though a model had
    been produced, and the run log would say nothing about it.
    """
    request = TrainingRequest(dataset=_dataset(), model_name=DEFAULT_MODEL)
    with pytest.raises(EstimatorUnavailableError):
        train_classifier(request)


def test_training_refusal_explains_the_rule_fallback_survives() -> None:
    """The refusal must say the application is unaffected.

    This is the reassurance an operator needs: declining to train is not an
    outage, because the rule-based detector still serves every classification.
    """
    request = TrainingRequest(dataset=_dataset(), model_name=DEFAULT_MODEL)
    with pytest.raises(EstimatorUnavailableError) as excinfo:
        train_classifier(request)
    assert "rule-based" in str(excinfo.value)


def test_training_error_is_an_estimator_error() -> None:
    """Callers must be able to catch the family."""
    request = TrainingRequest(dataset=_dataset(), model_name=DEFAULT_MODEL)
    with pytest.raises(TrainingError):
        train_classifier(request)


def test_no_dataset_means_no_training_data_error() -> None:
    """The no-data case must be nameable specifically.

    ``TrainingDataUnavailableError`` exists so a caller can tell "no benchmark
    downloaded yet" from "no estimator wired up" -- different problems needing
    different fixes.
    """
    assert issubclass(TrainingDataUnavailableError, TrainingError)


def test_confidence_floor_is_a_probability() -> None:
    """The fallback threshold must be expressible as one."""
    assert 0.0 < MODEL_CONFIDENCE_FLOOR < 1.0


def test_decision_range_accepts_the_bounds() -> None:
    """Both ends of the range must be valid confidences."""
    assert assert_decision_range(0.0) == 0.0
    assert assert_decision_range(1.0) == 1.0


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_decision_range_refuses_out_of_range(bad: float) -> None:
    """A confidence the predictions column would reject must be refused here."""
    with pytest.raises(ValueError):
        assert_decision_range(bad)


def test_run_summary_names_model_rows_and_classes() -> None:
    """A run log line must identify the model and its class distribution."""
    run = build_training_run(
        TrainingRequest(dataset=_dataset(), model_name=DEFAULT_MODEL),
        trained_at=datetime(2026, 1, 1),
    )
    summary = summarise_run(run)
    assert DEFAULT_MODEL in summary
    assert MODEL_VERSION in summary
    assert "Benign" in summary and "DoS" in summary


def test_classified_result_defaults_to_no_detail() -> None:
    """A result with nothing to add must still be constructible."""
    result = ClassifiedResult(
        attack_type=AttackType.BENIGN,
        confidence=0.5,
        source=ClassifierSource.MODEL,
    )
    assert result.detail == ""