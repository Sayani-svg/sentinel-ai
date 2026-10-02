"""Focused tests for the standalone ML config.

The ML config must resolve its own paths and settings without importing the
backend or requiring environment variables such as ``DATABASE_URL`` or
``SECRET_KEY``. These tests verify that the module-level constants and
:class:`MLPaths` behave as specified.
"""

from __future__ import annotations

from src.config import (
    BACKEND_DIR,
    BASE_DIR,
    CV_FOLDS,
    DATASET_DIR,
    DEFAULT_MODEL,
    DEFAULT_MODEL_FILENAME,
    ML_DIR,
    MODEL_DIR,
    MODEL_NAME,
    MODEL_PATH,
    MODEL_VERSION,
    PATHS,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
)


def test_ml_dir_is_the_ml_package() -> None:
    """The ML root must be this checkout's ``ml`` directory."""
    assert ML_DIR.name == "ml"
    assert ML_DIR.is_dir()


def test_backend_dir_is_the_backend_package() -> None:
    """The backend root must be this checkout's ``backend`` directory."""
    assert BACKEND_DIR.name == "backend"
    assert BACKEND_DIR.is_dir()


def test_layout_is_backend_and_ml_under_one_root() -> None:
    """Backend and ML must be siblings, not nested inside one another."""
    assert BACKEND_DIR.parent == ML_DIR.parent
    assert BASE_DIR == ML_DIR.parent


def test_paths_point_inside_the_checkout() -> None:
    """Derived directories must live under the repository, not beside it."""
    assert PATHS.dataset_dir == ML_DIR / "dataset"
    assert PATHS.model_dir == ML_DIR / "models"
    for path in (PATHS.ml_dir, BACKEND_DIR, DATASET_DIR, MODEL_DIR):
        assert BASE_DIR in path.parents or path == BASE_DIR


def test_ensure_model_dir_creates_on_demand() -> None:
    """A missing model directory must be creatable, for a fresh checkout."""
    assert PATHS.ensure_model_dir() == PATHS.model_dir
    assert PATHS.model_dir.is_dir()


def test_ensure_dataset_dir_creates_on_demand() -> None:
    """A missing dataset directory must be creatable, for a fresh checkout."""
    assert PATHS.ensure_dataset_dir() == PATHS.dataset_dir
    assert PATHS.dataset_dir.is_dir()


def test_model_filename_is_correct() -> None:
    """The artifact filename matches the expected default."""
    assert DEFAULT_MODEL_FILENAME == "best_model.pkl"
    assert MODEL_PATH == MODEL_DIR / DEFAULT_MODEL_FILENAME


def test_split_fractions_sum_below_one() -> None:
    """Train/validation/test splits must leave a share for training."""
    assert TEST_SIZE + VALIDATION_SIZE < 1.0
    assert TEST_SIZE == 0.2
    assert VALIDATION_SIZE == 0.1


def test_model_settings_are_set() -> None:
    """Model defaults are defined for standalone config."""
    assert CV_FOLDS == 5
    assert DEFAULT_MODEL == "XGBoost"
    assert MODEL_VERSION == "1.0.0"
    assert MODEL_NAME == "Sentinel AI Threat Classifier"
    assert RANDOM_STATE == 42


def test_paths_structure() -> None:
    """The MLPaths object exposes the expected directories."""
    assert PATHS.ml_dir == ML_DIR
    assert PATHS.dataset_dir == DATASET_DIR
    assert PATHS.model_dir == MODEL_DIR
    assert PATHS.outputs_dir == ML_DIR / "outputs"