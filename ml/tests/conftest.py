"""Import bootstrap for the ML tests.

The ML package imports the application's modules -- deliberately, so training and
serving read the same features and labels the detector does -- but the two trees
are not installed. ``ml/src`` must be importable as ``src`` and ``backend`` as
``app``, so both roots go on ``sys.path`` here rather than in every module.

The environment variables are the same ones ``backend/tests/conftest.py`` sets,
with the same values, because importing :mod:`app.services.log_parser` pulls in
the application settings, which require ``DATABASE_URL`` and ``SECRET_KEY``.
``setdefault`` throughout, so a real environment always wins and no test can be
made to pass by overriding a developer's own configuration.

No database is created. Nothing here connects to one; the ML tooling never does.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ML_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ML_DIR.parent / "backend"

for root in (BACKEND_DIR, ML_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg2://sentinel:sentinel@localhost:5432/sentinel_test",
)
os.environ.setdefault("SECRET_KEY", "test-only-secret-key-not-valid-in-production")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("ENABLE_FILE_LOGGING", "false")
os.environ.setdefault("BCRYPT_ROUNDS", "4")

from app.services.prediction_service import (  # noqa: E402 - needs sys.path above
    DetectionFeature,
    DetectionRecord,
)
from app.schemas.prediction import AttackType  # noqa: E402 - needs sys.path above

from src.feature_engineering import feature_names  # noqa: E402 - needs sys.path above


@pytest.fixture
def numeric_features() -> tuple[str, ...]:
    """Return the canonical feature names, in vector order.

    Returns:
        tuple[str, ...]: One name per position, from the ``DetectionFeature``
        enum with ``LABEL`` excluded.
    """
    return feature_names()


@pytest.fixture
def all_features() -> tuple[str, ...]:
    """Return every declared feature including the label column.

    Returns:
        tuple[str, ...]: The full ``DetectionFeature`` vocabulary.
    """
    return tuple(str(feature) for feature in DetectionFeature)


def _make_features(**overrides: float) -> dict[str, float]:
    """Build a complete feature mapping, with optional overrides.

    Every declared feature gets a distinct nonzero default so a test that mixes
    values up positionally fails visibly rather than accidentally. Tests still
    write the values that matter explicitly; this only supplies the rest.

    Args:
        **overrides: Canonical feature names to set explicitly.

    Returns:
        dict[str, float]: A mapping carrying every declared numeric feature.
    """
    defaults = {
        str(feature): 10.0 + index
        for index, feature in enumerate(feature_names())
    }
    defaults.update(overrides)
    return defaults


def _make_record(
    label: str | AttackType | None = None,
    *,
    drop: tuple[str, ...] = (),
    **overrides: float,
) -> DetectionRecord:
    """Build a record for tests.

    Args:
        label: Ground truth to attach, or ``None`` for an unlabelled record.
        drop: Declared features to omit, for testing refusal of absent features.
        **overrides: Canonical feature names to set explicitly.

    Returns:
        DetectionRecord: A record with every declared feature unless dropped.
    """
    features = _make_features(**overrides)
    for name in drop:
        features.pop(name, None)
    return DetectionRecord(
        features=features,
        label=label.value if isinstance(label, AttackType) else label,
    )


@pytest.fixture
def make_features():
    """Return the helper that builds a complete feature mapping.

    Returns:
        Callable[..., dict[str, float]]: Keyword overrides for canonical feature
        names, yielding every declared numeric feature.
    """
    return _make_features


@pytest.fixture
def make_record():
    """Return the helper that builds a :class:`DetectionRecord`.

    Returns:
        Callable[..., DetectionRecord]: A label, an optional tuple of features to
        omit, and keyword overrides for canonical feature names.
    """
    return _make_record