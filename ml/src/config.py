"""Paths and constants for the offline ML tooling.

Dataset preparation is an offline activity: it runs on an analyst's workstation
against a downloaded benchmark, with no API server and no database. Importing
:mod:`app.core.config` and calling :func:`~app.core.config.get_settings` would
demand ``DATABASE_URL`` and ``SECRET_KEY`` for a job that never touches either,
so this module resolves the directories itself instead.

The derivation is deliberately identical to
:func:`app.core.config._resolve_project_paths`, and
``tests/test_config.py`` asserts the two agree whenever the backend settings can
be constructed. If that assertion ever fails, the ML tooling and the running
application have drifted onto different directories, which is the kind of
disagreement that only surfaces once a model trained on one path is looked for
under another.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

#: Root of the ``ml`` package, i.e. ``<checkout>/ml``.
ML_DIR: Final[Path] = Path(__file__).resolve().parents[1]

#: Root of the ``backend`` package, i.e. ``<checkout>/backend``. Derived from the
#: ML directory rather than by counting parents, because counting parents from
#: this file lands on the checkout, not on ``backend``.
BACKEND_DIR: Final[Path] = ML_DIR.parent / "backend"

#: Root of the repository checkout.
BASE_DIR: Final[Path] = ML_DIR.parent

#: Where downloaded benchmark files are placed.
DATASET_DIR: Final[Path] = ML_DIR / "dataset"

#: Where trained artifacts are written.
MODEL_DIR: Final[Path] = ML_DIR / "models"

#: Default artifact filename, matching ``Settings.MODEL_PATH``.
DEFAULT_MODEL_FILENAME: Final[str] = "best_model.pkl"

#: Default artifact location, matching ``Settings.MODEL_PATH``.
MODEL_PATH: Final[Path] = MODEL_DIR / DEFAULT_MODEL_FILENAME

#: Seed used for every split and any estimator that takes one, matching
#: ``Settings.RANDOM_STATE``. Stated once so a split made during dataset
#: inspection is reproducible without consulting the application settings.
RANDOM_STATE: Final[int] = 42

#: Fraction of a dataset held back for validation, matching
#: ``Settings.VALIDATION_SIZE``.
VALIDATION_SIZE: Final[float] = 0.1

#: Fraction of a dataset held back for testing, matching ``Settings.TEST_SIZE``.
TEST_SIZE: Final[float] = 0.2

#: Number of cross-validation folds, matching ``Settings.CV_FOLDS``.
CV_FOLDS: Final[int] = 5

#: Estimator identifier used when a caller states none, matching
#: ``Settings.DEFAULT_MODEL``. Not a claim that this estimator is available: no
#: scientific stack is installed, so :func:`src.model_training.train_classifier`
#: refuses until one is.
DEFAULT_MODEL: Final[str] = "XGBoost"

#: Version label for the active artifact, matching ``Settings.MODEL_VERSION``.
MODEL_VERSION: Final[str] = "1.0.0"

#: Human-readable artifact name, matching ``Settings.MODEL_NAME``.
MODEL_NAME: Final[str] = "Sentinel AI Threat Classifier"


@dataclass(frozen=True, slots=True)
class MLPaths:
    """The directories the ML tooling reads from and writes to.

    Resolved once at import. Tests that need a different root construct their own
    instance rather than mutating the module-level one, so two tests cannot
    disagree about where the dataset lives.
    """

    ml_dir: Path
    dataset_dir: Path
    model_dir: Path
    outputs_dir: Path

    def ensure_model_dir(self) -> Path:
        """Create the model directory if it is absent and return it.

        A fresh checkout has ``ml/models`` as a tracked but empty directory, and a
        directory can be missing entirely once someone cleans the tree, so the
        artifact writer creates it on demand rather than assuming it exists.

        Returns:
            Path: The existing model directory.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        return self.model_dir

    def ensure_dataset_dir(self) -> Path:
        """Create the dataset directory if it is absent and return it.

        Returns:
            Path: The existing dataset directory.
        """
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        return self.dataset_dir


#: The paths this checkout resolves to.
PATHS: Final[MLPaths] = MLPaths(
    ml_dir=ML_DIR,
    dataset_dir=DATASET_DIR,
    model_dir=MODEL_DIR,
    outputs_dir=ML_DIR / "outputs",
)