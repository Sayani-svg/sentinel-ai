"""The ML package must resolve the same directories the application does.

These are the tests that fail loudly if the two ever drift. A model trained from
``ml/dataset`` and served from ``backend``'s view of the world has to agree about
where those files are, and the only moment that is cheap to notice is before a
model exists. Once one does, a path mistake becomes a silent "no artifact found"
at serving time.
"""

from __future__ import annotations

from src.config import BACKEND_DIR, BASE_DIR, DATASET_DIR, ML_DIR, MODEL_DIR, PATHS


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
    """Derived directories must live under the repository, not beside it.

    Guards the specific mistake of counting parents from ``ml/src`` and landing
    one level too high, which puts the dataset directory outside the repository.
    """
    assert PATHS.dataset_dir == ML_DIR / "dataset"
    assert PATHS.model_dir == ML_DIR / "models"
    for path in (PATHS.ml_dir, BACKEND_DIR, DATASET_DIR, MODEL_DIR):
        assert BASE_DIR in path.parents or path == BASE_DIR


def test_agrees_with_application_settings() -> None:
    """ML paths must match the ones the running application resolves.

    The application settings are imported only for this comparison. They are not
    needed to compute the ML paths -- that independence is the point of
    :mod:`src.config` -- but where both are available they must agree.
    """
    from app.core.config import _resolve_project_paths

    expected = _resolve_project_paths()

    assert str(ML_DIR) == str(expected["ML_DIR"])
    assert str(DATASET_DIR) == str(expected["DATASET_DIR"])
    assert str(MODEL_DIR) == str(expected["MODEL_DIR"])


def test_ensure_model_dir_creates_on_demand() -> None:
    """A missing model directory must be creatable, for a fresh checkout."""
    assert PATHS.ensure_model_dir() == PATHS.model_dir
    assert PATHS.model_dir.is_dir()


def test_model_filename_matches_application_setting() -> None:
    """The artifact filename must match ``Settings.MODEL_PATH``.

    The application looks in one specific place. A loader defaulting elsewhere
    would report the model as missing while the file sat on disk.
    """
    from app.core.config import Settings

    from src.config import DEFAULT_MODEL_FILENAME, MODEL_PATH

    assert DEFAULT_MODEL_FILENAME == "best_model.pkl"
    assert MODEL_PATH == Settings().MODEL_PATH


def test_split_fractions_sum_below_one() -> None:
    """Train/validation/test splits must leave a share for training.

    Mirrors the application's own validator. Stating the invariant here means a
    future default that breaks it is caught in this package's tests rather than
    whenever a training run first tries to split anything.
    """
    from app.core.config import Settings

    from src.config import TEST_SIZE, VALIDATION_SIZE

    assert TEST_SIZE + VALIDATION_SIZE < 1.0
    assert (TEST_SIZE, VALIDATION_SIZE) == (Settings().TEST_SIZE, Settings().VALIDATION_SIZE)