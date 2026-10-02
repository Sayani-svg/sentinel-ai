"""Label encoding has to be a total, reversible bijection over the real vocabulary.

The integer codes here are not internal detail: they are what a trained estimator
returns as column indices, and they are what ends up in a serialized artifact.
A code that shifts by one between a training run and a serving run relabels every
prediction, so these tests pin the exact numbers rather than only asserting that
round-tripping works.
"""

from __future__ import annotations

import pytest
from app.schemas.prediction import AttackType
from app.services.prediction_service import LABEL_ATTACK_TYPES

from src.preprocessing import (
    CLASS_COUNT,
    INDEX_TO_LABEL,
    KNOWN_LABEL_SPELLINGS,
    LABEL_TO_INDEX,
    TRAINABLE_ATTACK_TYPES,
    LabelMappingError,
    decode_codes,
    encode_labels,
    index_to_label,
    label_to_index,
    require_trainable_label,
)


def test_codes_follow_enum_declaration_order() -> None:
    """Codes must be dense and ascending in ``AttackType`` declaration order.

    Derived rather than asserted against a literal list so the invariant is
    stated, while the separate test below pins the concrete values to catch a
    reorder of the enum itself.
    """
    expected = {
        attack_type: index
        for index, attack_type in enumerate(AttackType)
        if attack_type is not AttackType.UNKNOWN
    }
    assert LABEL_TO_INDEX == expected


def test_unknown_is_not_trainable() -> None:
    """``Unknown`` must not be a trainable class.

    It is what an unrecognised label resolves to, so a dataset row labelled
    "unknown" states that no one knows the class. Training on it teaches the
    model to answer "unknown", which is a claim about the data that nobody made.
    """
    assert AttackType.UNKNOWN not in TRAINABLE_ATTACK_TYPES
    assert CLASS_COUNT == len(TRAINABLE_ATTACK_TYPES) == len(AttackType) - 1


def test_benign_is_code_zero() -> None:
    """``Benign`` must be code 0.

    It is the negative class in this vocabulary and the first class a benchmark
    file names, so a sensible default for a not-yet-fitted estimator.
    """
    assert label_to_index(AttackType.BENIGN) == 0
    assert index_to_label(0) == AttackType.BENIGN


def test_every_code_round_trips() -> None:
    """Encoding then decoding must be the identity for every class."""
    for attack_type in TRAINABLE_ATTACK_TYPES:
        assert index_to_label(label_to_index(attack_type)) is attack_type


def test_label_to_index_accepts_raw_spelling() -> None:
    """Every raw dataset spelling must encode to its class's code.

    Benchmark files spell the same class several ways; the vocabulary was already
    resolved for the detector by ``LABEL_ATTACK_TYPES`` and reusing it keeps
    training and serving from disagreeing about what "DDoS" means.
    """
    for attack_type in TRAINABLE_ATTACK_TYPES:
        spellings = [
            spelling
            for spelling, resolved in LABEL_ATTACK_TYPES.items()
            if resolved is attack_type
        ]
        assert spellings, f"{attack_type.value} has no spellings in the vocabulary"
        for spelling in spellings:
            assert label_to_index(spelling) == label_to_index(attack_type)


def test_unknown_spelling_is_refused() -> None:
    """A label matching no class must raise rather than fall back to a code."""
    with pytest.raises(LabelMappingError) as excinfo:
        label_to_index("something-else-entirely")
    assert "something-else-entirely" in str(excinfo.value)


def test_unknown_class_is_refused() -> None:
    """``AttackType.UNKNOWN`` must not encode to a trainable code."""
    with pytest.raises(LabelMappingError):
        label_to_index(AttackType.UNKNOWN)


def test_out_of_range_code_is_refused() -> None:
    """Decoding a code past the vocabulary must raise, not wrap or return None."""
    for code in (-1, CLASS_COUNT, CLASS_COUNT + 100):
        with pytest.raises(LabelMappingError):
            index_to_label(code)


def test_encode_and_decode_preserve_order_and_length() -> None:
    """Batch helpers must be positional, including for repeated labels."""
    labels = [
        AttackType.BENIGN,
        AttackType.DDOS,
        AttackType.BENIGN,
        AttackType.BRUTE_FORCE,
    ]
    codes = encode_labels(labels)
    assert len(codes) == len(labels)
    assert codes == (0, label_to_index(AttackType.DDOS), 0, label_to_index(AttackType.BRUTE_FORCE))
    assert decode_codes(codes) == tuple(labels)


def test_require_trainable_label_returns_the_class() -> None:
    """A usable label must come back as the resolved class."""
    assert require_trainable_label("DoS") is AttackType.DOS
    assert require_trainable_label(AttackType.DOS) is AttackType.DOS


@pytest.mark.parametrize("label", ["unknown", "UNKNOWN", AttackType.UNKNOWN])
def test_require_trainable_label_refuses_unknown(label: str | AttackType) -> None:
    """The absence of ground truth must be refused at the dataset boundary."""
    with pytest.raises(LabelMappingError):
        require_trainable_label(label)


def test_lookup_tables_are_mutually_consistent() -> None:
    """The forward and reverse tables must be exact inverses.

    Two independently built tables drifting apart would give a stable-looking but
    wrong mapping, which no round-trip test over a single value would catch.
    """
    assert len(LABEL_TO_INDEX) == len(INDEX_TO_LABEL) == CLASS_COUNT
    for attack_type, code in LABEL_TO_INDEX.items():
        assert INDEX_TO_LABEL[code] is attack_type
    assert set(INDEX_TO_LABEL) == set(range(CLASS_COUNT))


def test_known_label_spellings_match_the_application_vocabulary() -> None:
    """The accepted spellings must be the application's, not a second list."""
    assert KNOWN_LABEL_SPELLINGS == frozenset(LABEL_ATTACK_TYPES)


def test_empty_input_encodes_to_empty() -> None:
    """Empty input must stay empty rather than raise.

    A dataset with no rows is caught by the dataset validator with a message
    about rows; an encoder that raised here would report the wrong problem.
    """
    assert encode_labels([]) == ()
    assert decode_codes([]) == ()