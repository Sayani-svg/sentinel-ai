"""Serving must degrade to the rules visibly, and must never serve a broken artifact.

The estimator below is a hand-written test double. It exists to exercise the
aggregation, the metadata checks and the fallback threshold without inventing
training data or a model: it returns numbers this test states outright, so
nothing about it is a claim about how a real classifier performs.

The fallback tests matter most. A deployment that quietly answers from thresholds
while its operators believe a model is running is the failure this module is
built to make impossible, so every path asserts on :attr:`ClassifiedResult.source`.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from app.schemas.prediction import AttackType
from app.services.log_parser import parse_csv_log
from app.services.prediction_service import DetectionInput, detect_attack

from src.feature_engineering import (
    FeatureOrderError,
    feature_names,
    to_feature_space,
)
from src.inference import (
    ArtifactContractError,
    ArtifactMetadata,
    FallbackClassifier,
    InferenceError,
    ModelArtifact,
    ModelArtifactUnavailableError,
    ModelClassifier,
    SerializationUnavailableError,
    load_artifact,
    select_classifier,
)
from src.model_training import (
    ClassifierSource,
    ClassifiedResult,
    RuleBasedClassifier,
)

CLASSES: tuple[AttackType, ...] = (AttackType.BENIGN, AttackType.DOS)


class StubEstimator:
    """A test double returning probabilities this test states.

    Attributes:
        classes_: Class order its probability columns correspond to.
        rows: One probability row per input row, consumed in order.
    """

    def __init__(self, rows: list[list[float]]) -> None:
        """Record the class order and the rows to return.

        Args:
            rows: Probability rows to return, one per scoring call.
        """
        self.classes_ = CLASSES
        self._rows = rows

    def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
        """Return the pre-stated probability rows.

        Args:
            features: The row-major matrix. Unused; the double does no arithmetic.

        Returns:
            list[list[float]]: One row per call, cycling through ``self._rows``.
        """
        return self._rows


def _detection_input(csv: str) -> DetectionInput:
    """Return a prepared detection input from CSV text.

    Args:
        csv: The CSV text, header included.

    Returns:
        DetectionInput: The records in the canonical feature space.
    """
    return to_feature_space(parse_csv_log(io.StringIO(csv)))


def _two_record_file() -> DetectionInput:
    """Return a file with two records and a mapped feature set.

    Returns:
        DetectionInput: Two rows carrying every declared feature.
    """
    return _detection_input(
        "Flow Duration,Total Length,Total Packets,Avg Packet Size,Packets Per Second\n"
        "10,1000,5,200,1\n20,2000,10,200,2\n"
    )


def _metadata(feature_order: tuple[str, ...] | None = None) -> ArtifactMetadata:
    """Return valid artifact metadata.

    Args:
        feature_order: Feature order to record, or ``None`` for this build's.

    Returns:
        ArtifactMetadata: Metadata naming ``model_name`` and ``version``.
    """
    return ArtifactMetadata(
        model_name="StubEstimator",
        version="0.0.0-test",
        feature_order=feature_names() if feature_order is None else feature_order,
        classes=CLASSES,
    )


def _model(rows: list[list[float]], *, confidence_floor: float = 0.5) -> ModelClassifier:
    """Return a :class:`ModelClassifier` over the stub estimator.

    Args:
        rows: Probability rows the double should return.
        confidence_floor: Threshold below which the fallback decides.

    Returns:
        ModelClassifier: A validated model path.
    """
    return ModelClassifier(
        ModelArtifact(estimator=StubEstimator(rows), metadata=_metadata()),
        confidence_floor=confidence_floor,
    )


def test_missing_artifact_names_the_path_looked_at(tmp_path: Path) -> None:
    """A missing artifact must name where it was expected.

    "Model not loaded" without a location is the least actionable message in the
    codebase: nobody can tell whether to train, fix the path, or give up.
    """
    target = tmp_path / "absent.pkl"
    with pytest.raises(ModelArtifactUnavailableError) as excinfo:
        load_artifact(target)

    assert excinfo.value.path == target
    assert str(target) in str(excinfo.value)


def test_artifact_error_says_the_fallback_still_serves(tmp_path: Path) -> None:
    """The refusal must say the application is not broken."""
    with pytest.raises(ModelArtifactUnavailableError) as excinfo:
        load_artifact(tmp_path / "absent.pkl")
    assert "rule-based" in str(excinfo.value)


def test_empty_artifact_is_refused(tmp_path: Path) -> None:
    """A zero-byte placeholder must not be treated as a model.

    An empty file satisfies a presence check, so serving it would appear to work
    while classifying nothing.
    """
    target = tmp_path / "best_model.pkl"
    target.write_bytes(b"")

    with pytest.raises(ModelArtifactUnavailableError) as excinfo:
        load_artifact(target)
    assert "empty" in str(excinfo.value)


def test_directory_is_refused(tmp_path: Path) -> None:
    """A directory where an artifact is expected must be refused."""
    target = tmp_path / "best_model.pkl"
    target.mkdir()

    with pytest.raises(ModelArtifactUnavailableError):
        load_artifact(target)


def test_present_artifact_without_a_serializer_is_refused(tmp_path: Path) -> None:
    """A real file must still be refused, naming the missing serializer.

    joblib and pickle are both absent, so there is no honest way to return an
    estimator in this build.
    """
    target = tmp_path / "best_model.pkl"
    target.write_bytes(b"not a real artifact")

    with pytest.raises(SerializationUnavailableError) as excinfo:
        load_artifact(target)
    assert "serializer" in str(excinfo.value)


def test_serialization_error_is_an_artifact_error(tmp_path: Path) -> None:
    """Callers may catch the family when deciding to fall back."""
    target = tmp_path / "best_model.pkl"
    target.write_bytes(b"x")
    with pytest.raises(ModelArtifactUnavailableError):
        load_artifact(target)


def test_select_returns_rules_when_no_artifact(tmp_path: Path) -> None:
    """Selection must succeed, not raise, when there is no model to load."""
    classifier = select_classifier(tmp_path / "absent.pkl")

    assert isinstance(classifier, FallbackClassifier)
    assert not classifier.has_model
    assert classifier.source is ClassifierSource.RULES


def test_select_defaults_to_the_rule_based_detector(tmp_path: Path) -> None:
    """The default fallback must be today's detector, not a null object."""
    classifier = select_classifier(tmp_path / "absent.pkl")
    assert classifier.feature_order == feature_names()


def test_selected_fallback_reproduces_existing_behaviour(tmp_path: Path) -> None:
    """Serving through the selection path must not change any verdict.

    This is the test that makes the fallback safe to ship: the classification the
    application produces today is byte-for-byte the one it will produce with this
    module in place.
    """
    detection_input = _detection_input(
        "Flow Duration,Total Length,Total Packets,Avg Packet Size,Packets Per Second\n"
        "100,5000000,4000,1250,4000\n"
    )
    expected = detect_attack(detection_input)

    result = select_classifier(tmp_path / "absent.pkl").classify(detection_input)

    assert result.attack_type is expected.attack_type
    assert result.confidence == expected.confidence
    assert result.source is ClassifierSource.RULES


def test_confident_model_result_is_stamped_as_model() -> None:
    """A confident model decision must be attributable to the model."""
    model = _model([[0.9, 0.1], [0.8, 0.2]])
    result = model.classify(_two_record_file())

    assert result.attack_type is AttackType.BENIGN
    assert result.source is ClassifierSource.MODEL


def test_model_confidence_is_the_mean_winning_mass() -> None:
    """Confidence must be the winning class's share of probability mass.

    Both rows put 0.9 and 0.8 on ``Benign``, so its mean is 0.85.
    """
    result = _model([[0.9, 0.1], [0.8, 0.2]]).classify(_two_record_file())
    assert result.confidence == pytest.approx(0.85)


def test_model_aggregates_per_record_scores_into_one_verdict() -> None:
    """The model path must decide per file, like the rule path does."""
    result = _model([[0.1, 0.9], [0.2, 0.8]]).classify(_two_record_file())
    assert result.attack_type is AttackType.DOS


def test_model_rows_are_scored_in_record_order() -> None:
    """Per-record scores must align with the records they came from."""
    model = _model([[0.9, 0.1], [0.1, 0.9]])
    scores = model.score_rows(_two_record_file())
    assert [attack_type for attack_type, _ in scores] == [AttackType.BENIGN, AttackType.DOS]


def test_model_decision_is_deterministic_under_ties() -> None:
    """Equal mass must resolve by class code, so repeat runs agree.

    Without a tie-break the winner would depend on dictionary order, and a
    retrained artifact could silently change its verdicts.
    """
    result = _model([[0.5, 0.5], [0.5, 0.5]]).classify(_two_record_file())
    assert result.attack_type is AttackType.BENIGN


def test_model_result_names_its_artifact() -> None:
    """A model verdict must be traceable to a version."""
    result = _model([[0.9, 0.1], [0.9, 0.1]]).classify(_two_record_file())
    assert "StubEstimator" in result.detail


def test_empty_file_is_refused_by_the_model_path() -> None:
    """A file with no records must not produce a verdict.

    The parser already refuses a header-only file, so this is defence in depth:
    the classifier must not hand an empty matrix to an estimator, whose behaviour
    on zero rows is not something to depend on.
    """
    from app.services.prediction_service import DetectionInput

    with pytest.raises(InferenceError):
        _model([]).classify(DetectionInput(records=()))


def test_fallback_defers_to_rules_for_an_unconfident_model() -> None:
    """A model below the floor must hand over to the rules.

    Two rows split evenly leave the winning class at 0.5, which the 0.6 floor
    rejects.
    """
    classifier = FallbackClassifier(
        _model([[0.5, 0.5], [0.5, 0.5]], confidence_floor=0.6),
        RuleBasedClassifier(),
        confidence_floor=0.6,
    )
    result = classifier.classify(_two_record_file())
    assert result.source is ClassifierSource.RULES


def test_fallback_keeps_the_model_when_it_is_confident() -> None:
    """A confident model must not be second-guessed."""
    classifier = FallbackClassifier(
        _model([[0.99, 0.01], [0.99, 0.01]]),
        RuleBasedClassifier(),
    )
    result = classifier.classify(_two_record_file())
    assert result.source is ClassifierSource.MODEL


def test_fallback_explains_the_deferral() -> None:
    """A deferred verdict must say why, and what the model thought.

    This is what makes the deferral auditable instead of mysterious.
    """
    classifier = FallbackClassifier(
        _model([[0.5, 0.5], [0.5, 0.5]], confidence_floor=0.6),
        RuleBasedClassifier(),
        confidence_floor=0.6,
    )
    detail = classifier.classify(_two_record_file()).detail

    assert "below floor" in detail
    assert "Benign" in detail or "DoS" in detail


def test_fallback_uses_the_rules_entirely_when_no_model_exists() -> None:
    """With no model, the rules must decide and say so."""
    classifier = FallbackClassifier(None, RuleBasedClassifier())
    assert not classifier.has_model
    assert classifier.classify(_two_record_file()).source is ClassifierSource.RULES


def test_fallback_prefers_the_model_feature_order() -> None:
    """With a model loaded, its positional contract must be the one reported."""
    classifier = FallbackClassifier(
        _model([[0.9, 0.1], [0.9, 0.1]]), RuleBasedClassifier()
    )
    assert classifier.feature_order == feature_names()


def test_metadata_rejects_a_feature_order_it_cannot_serve() -> None:
    """An artifact trained against another order must be refused.

    Positional features produce confident wrong answers rather than errors, so
    this check is the only thing standing between a reordered artifact and a
    stream of plausible nonsense.
    """
    metadata = _metadata(feature_order=tuple(reversed(feature_names())))
    with pytest.raises(FeatureOrderError):
        metadata.validate()


def test_metadata_rejects_unknown_classes() -> None:
    """An artifact predicting classes this build lacks must be refused."""
    metadata = ArtifactMetadata(
        model_name="StubEstimator",
        version="0.0.0-test",
        feature_order=feature_names(),
        classes=(AttackType.BENIGN, AttackType.UNKNOWN),
    )
    with pytest.raises(ArtifactContractError):
        metadata.validate()


def test_artifact_rejects_inconsistent_class_order() -> None:
    """Estimator and metadata must agree on which column is which label.

    Disagreement means probability column 0 is labelled as class 1, which is
    wrong for every prediction with no visible symptom.
    """

    class Mislabelled(StubEstimator):
        """A double whose class order disagrees with its metadata."""

        def __init__(self) -> None:
            super().__init__([[0.5, 0.5]])
            self.classes_ = (AttackType.DOS, AttackType.BENIGN)

    artifact = ModelArtifact(estimator=Mislabelled(), metadata=_metadata())
    with pytest.raises(ArtifactContractError):
        artifact.validate()


def test_model_classifier_validates_on_construction() -> None:
    """A broken artifact must fail when wrapped, not at the first request."""
    artifact = ModelArtifact(
        estimator=StubEstimator([[0.5, 0.5]]),
        metadata=_metadata(feature_order=tuple(reversed(feature_names()))),
    )
    with pytest.raises(FeatureOrderError):
        ModelClassifier(artifact)


def test_serving_agrees_with_the_training_side_encoding() -> None:
    """Serving's class order must be the one training encoded against.

    ``ModelClassifier`` reads probability columns positionally against
    ``ArtifactMetadata.classes``, so those must be in integer-code order -- the
    same order :func:`src.preprocessing.label_to_index` produces.
    """
    from src.preprocessing import index_to_label, label_to_index

    assert tuple(label_to_index(c) for c in CLASSES) == tuple(
        sorted(label_to_index(c) for c in CLASSES)
    )
    assert tuple(index_to_label(label_to_index(c)) for c in CLASSES) == CLASSES


def test_result_source_distinguishes_the_two_paths() -> None:
    """The two sources must be distinct values, or the stamp means nothing."""
    assert ClassifierSource.MODEL != ClassifierSource.RULES
    assert str(ClassifierSource.MODEL) == "model"
    assert str(ClassifierSource.RULES) == "rules"


def test_classified_result_is_immutable() -> None:
    """A result stamped with its provenance must not be re-stamped later."""
    result = ClassifiedResult(
        attack_type=AttackType.BENIGN,
        confidence=0.9,
        source=ClassifierSource.RULES,
    )
    with pytest.raises(AttributeError):
        result.source = ClassifierSource.MODEL  # type: ignore[misc]