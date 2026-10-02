"""Label encoding for the threat classes the detector reports.

The class vocabulary is not defined here. It is
:class:`app.schemas.prediction.AttackType`, the same nine values the
``predictions.attack_type`` column accepts, and the spellings a benchmark file
uses are resolved by
:func:`app.services.prediction_service.attack_type_for_label`. Declaring a
parallel set of classes here would be the fastest way to end up training a model
whose output the API cannot store.

**``Unknown`` is not a trainable class.** The enum documents it as the honest
answer for a log carrying no recognisable signal, never a synonym for benign. A
dataset row labelled ``Unknown`` therefore carries no ground truth: training on
it would teach the model to reproduce the detector's own uncertainty. Rows so
labelled are refused by :func:`require_trainable_label` rather than silently
folded into another class.

**Integer codes are derived, never assigned.** They come from
:class:`AttackType` declaration order, so a code can only change if the enum
itself changes, and the mapping is a bijection by construction. A hand-kept
``{0: "DoS", 1: "DDoS"}`` table is the kind of thing that survives a class rename
and then quietly mislabels every prediction made in between.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

from app.schemas.prediction import AttackType
from app.services.prediction_service import (
    LABEL_ATTACK_TYPES,
    attack_type_for_label,
    normalize_label,
)

#: Classes a model may be trained to predict, in enum declaration order.
#: ``Unknown`` is excluded; see the module docstring.
TRAINABLE_ATTACK_TYPES: Final[tuple[AttackType, ...]] = tuple(
    attack_type for attack_type in AttackType if attack_type is not AttackType.UNKNOWN
)

#: Attack type to integer code. Derived from declaration order.
LABEL_TO_INDEX: Final[Mapping[AttackType, int]] = {
    attack_type: index for index, attack_type in enumerate(TRAINABLE_ATTACK_TYPES)
}

#: Integer code back to attack type, the exact inverse of :data:`LABEL_TO_INDEX`.
INDEX_TO_LABEL: Final[Mapping[int, AttackType]] = {
    index: attack_type for attack_type, index in LABEL_TO_INDEX.items()
}

#: Number of distinct classes a trained model can output.
CLASS_COUNT: Final[int] = len(TRAINABLE_ATTACK_TYPES)

#: Every spelling :func:`attack_type_for_label` can resolve, for reporting.
KNOWN_LABEL_SPELLINGS: Final[frozenset[str]] = frozenset(LABEL_ATTACK_TYPES)


class LabelMappingError(ValueError):
    """Raised when a label cannot be encoded or a code cannot be decoded."""


def label_to_index(attack_type: AttackType | str) -> int:
    """Return the integer code for a threat class.

    A raw string is resolved through the benchmark vocabulary first, so ``"DoS
    Hulk"`` and ``AttackType.DOS`` both yield the same code.

    Args:
        attack_type: The class, as an enum member or a label as spelled in a
            dataset file.

    Returns:
        int: The class's code, in ``0..CLASS_COUNT - 1``.

    Raises:
        LabelMappingError: If the value names a class outside
            :data:`TRAINABLE_ATTACK_TYPES`, which includes
            :attr:`~app.schemas.prediction.AttackType.UNKNOWN` and any spelling
            the benchmark vocabulary does not recognise.
    """
    resolved = (
        attack_type
        if isinstance(attack_type, AttackType)
        else attack_type_for_label(attack_type)
    )

    code = LABEL_TO_INDEX.get(resolved)
    if code is None:
        # An unrecognised spelling resolves to UNKNOWN, so echoing only the
        # resolved value would report "Unknown" and hide what the file actually
        # said -- the part an operator needs in order to fix the file.
        stated = resolved.value if isinstance(attack_type, AttackType) else str(attack_type)
        raise LabelMappingError(
            f"{stated!r} is not a trainable class. "
            f"Trainable classes: "
            f"{', '.join(a.value for a in TRAINABLE_ATTACK_TYPES)}."
        )
    return code


def index_to_label(code: int) -> AttackType:
    """Return the threat class an integer code names.

    Args:
        code: The class code, as produced by :func:`label_to_index`.

    Returns:
        AttackType: The class the code names.

    Raises:
        LabelMappingError: If the code is not in ``0..CLASS_COUNT - 1``.
    """
    attack_type = INDEX_TO_LABEL.get(code)
    if attack_type is None:
        raise LabelMappingError(
            f"{code!r} is not a class code. Codes run "
            f"0..{CLASS_COUNT - 1}."
        )
    return attack_type


def require_trainable_label(label: str | AttackType) -> AttackType:
    """Resolve a dataset label to a class that can be trained on.

    Args:
        label: The label as spelled in the dataset file, or an enum member.

    Returns:
        AttackType: The resolved class.

    Raises:
        LabelMappingError: If the label resolves to
            :attr:`~app.schemas.prediction.AttackType.UNKNOWN`, meaning the file
            used a spelling the vocabulary does not recognise or stated no class
            at all. Guessing a class for it would fabricate ground truth.
    """
    resolved = (
        label if isinstance(label, AttackType) else attack_type_for_label(label)
    )
    if resolved is AttackType.UNKNOWN:
        raise LabelMappingError(
            f"Label {normalize_label(str(label))!r} does not name a known class. "
            f"Recognised spellings: "
            f"{', '.join(sorted(KNOWN_LABEL_SPELLINGS))}."
        )
    return resolved


def encode_labels(labels: Iterable[str | AttackType]) -> tuple[int, ...]:
    """Encode a sequence of dataset labels into class codes.

    Args:
        labels: Labels in dataset row order.

    Returns:
        tuple[int, ...]: One code per label, positionally aligned with the input.

    Raises:
        LabelMappingError: If any label is not a trainable class.
    """
    return tuple(label_to_index(label) for label in labels)


def decode_codes(codes: Iterable[int]) -> tuple[AttackType, ...]:
    """Decode class codes back into threat classes.

    Args:
        codes: Codes as produced by :func:`encode_labels`.

    Returns:
        tuple[AttackType, ...]: One class per code.

    Raises:
        LabelMappingError: If any code is out of range.
    """
    return tuple(index_to_label(code) for code in codes)