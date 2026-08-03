"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

EnvironmentName = Literal["development", "testing", "staging", "production"]

SUPPORTED_JWT_ALGORITHMS: frozenset[str] = frozenset(
    {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512"}
)

SUPPORTED_DATABASE_SCHEMES: frozenset[str] = frozenset(
    {"postgresql", "postgresql+psycopg2", "postgresql+asyncpg", "postgres"}
)


def _resolve_project_paths() -> dict[str, Path]:
    """Resolve platform-independent project directory paths once at import time.

    Returns:
        dict[str, Path]: Mapping of directory setting names to absolute paths.
    """
    backend_dir = Path(__file__).resolve().parent.parent.parent
    base_dir = backend_dir.parent
    ml_dir = base_dir / "ml"
    return {
        "BASE_DIR": base_dir,
        "BACKEND_DIR": backend_dir,
        "ML_DIR": ml_dir,
        "DATASET_DIR": ml_dir / "dataset",
        "MODEL_DIR": ml_dir / "models",
        "LOG_DIR": base_dir / "logs",
        "REPORT_DIR": base_dir / "reports",
        "UPLOAD_DIR": base_dir / "uploads",
    }


_PROJECT_PATHS: dict[str, Path] = _resolve_project_paths()


class Settings(BaseSettings):
    """Application settings sourced from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_NAME: str = Field(
        default="Sentinel AI",
        min_length=1,
        max_length=128,
        description="Human-readable application name.",
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        min_length=1,
        max_length=32,
        description="Semantic application version string.",
    )
    ENVIRONMENT: EnvironmentName = Field(
        default="development",
        description="Deployment environment name.",
    )
    API_V1_PREFIX: str = Field(
        default="/api/v1",
        min_length=1,
        description="URL prefix for version 1 API routes.",
    )
    API_TITLE: str = Field(
        default="Sentinel AI API",
        min_length=1,
        description="OpenAPI documentation title.",
    )
    API_DESCRIPTION: str = Field(
        default="AI Powered Security Operations Center",
        min_length=1,
        description="OpenAPI documentation description.",
    )
    BASE_DIR: Path = Field(
        default=_PROJECT_PATHS["BASE_DIR"],
        description="Repository root directory.",
    )
    BACKEND_DIR: Path = Field(
        default=_PROJECT_PATHS["BACKEND_DIR"],
        description="Backend application root directory.",
    )
    ML_DIR: Path = Field(
        default=_PROJECT_PATHS["ML_DIR"],
        description="Machine learning module root directory.",
    )
    DATASET_DIR: Path = Field(
        default=_PROJECT_PATHS["DATASET_DIR"],
        description="Directory containing training and evaluation datasets.",
    )
    MODEL_DIR: Path = Field(
        default=_PROJECT_PATHS["MODEL_DIR"],
        description="Directory containing serialized model artifacts.",
    )
    LOG_DIR: Path = Field(
        default=_PROJECT_PATHS["LOG_DIR"],
        description="Directory for application log files.",
    )
    REPORT_DIR: Path = Field(
        default=_PROJECT_PATHS["REPORT_DIR"],
        description="Directory for generated security reports.",
    )
    UPLOAD_DIR: Path = Field(
        default=_PROJECT_PATHS["UPLOAD_DIR"],
        description="Directory for uploaded log files.",
    )
    DATABASE_URL: str = Field(
        ...,
        min_length=1,
        description="SQLAlchemy-compatible PostgreSQL connection URL.",
    )
    DATABASE_ECHO: bool = Field(
        default=False,
        description="Enable SQLAlchemy SQL statement logging.",
    )
    DATABASE_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        description="Number of persistent database connections in the pool.",
    )
    DATABASE_MAX_OVERFLOW: int = Field(
        default=20,
        ge=0,
        description="Maximum number of overflow database connections.",
    )
    DATABASE_POOL_TIMEOUT: int = Field(
        default=30,
        ge=1,
        description="Seconds to wait for an available database connection.",
    )
    DATABASE_POOL_RECYCLE: int = Field(
        default=1800,
        ge=60,
        description="Seconds after which pooled connections are recycled.",
    )
    SECRET_KEY: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("SECRET_KEY", "JWT_SECRET_KEY"),
        description="Secret key used to sign and verify JWT access tokens.",
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT signing algorithm.",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=1,
        le=24 * 60,
        description="Access token lifetime in minutes.",
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        ge=1,
        le=365,
        description="Refresh token lifetime in days.",
    )
    PASSWORD_MIN_LENGTH: int = Field(
        default=12,
        ge=8,
        le=128,
        description="Minimum allowed password length for user accounts.",
    )
    BCRYPT_ROUNDS: int = Field(
        default=12,
        ge=4,
        le=31,
        description="bcrypt work factor for password hashing.",
    )
    MODEL_PATH: Path = Field(
        default=_PROJECT_PATHS["MODEL_DIR"] / "best_model.pkl",
        description="Filesystem path to the serialized ML model artifact.",
    )
    RANDOM_STATE: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducible machine learning workflows.",
    )
    TEST_SIZE: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Fraction of data reserved for the test split.",
    )
    VALIDATION_SIZE: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Fraction of data reserved for validation during training.",
    )
    CV_FOLDS: int = Field(
        default=5,
        ge=2,
        description="Number of cross-validation folds.",
    )
    DEFAULT_MODEL: str = Field(
        default="XGBoost",
        min_length=1,
        description="Default machine learning model identifier.",
    )
    MODEL_VERSION: str = Field(
        default="1.0.0",
        min_length=1,
        description="Version label for the active model artifact.",
    )
    MODEL_NAME: str = Field(
        default="Sentinel AI Threat Classifier",
        min_length=1,
        description="Human-readable name for the active model.",
    )
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=100,
        gt=0,
        description="Maximum allowed upload size in megabytes.",
    )
    SUPPORTED_UPLOAD_EXTENSIONS: list[str] = Field(
        default_factory=lambda: [".csv", ".json"],
        description="Permitted file extensions for log uploads.",
    )
    LOG_LEVEL: LogLevel = Field(
        default="INFO",
        description="Root logging level for the application.",
    )
    LOG_FORMAT: str = Field(
        default="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        min_length=1,
        description="Python logging format string.",
    )
    LOG_FILE_NAME: str = Field(
        default="sentinel-ai.log",
        min_length=1,
        description="Primary application log file name.",
    )
    ENABLE_FILE_LOGGING: bool = Field(
        default=True,
        description="Persist application logs to the configured log directory.",
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode for verbose errors and relaxed checks.",
    )
    ENABLE_SHAP: bool = Field(
        default=True,
        description="Enable SHAP-based prediction explainability.",
    )
    ENABLE_REPORTS: bool = Field(
        default=True,
        description="Enable report generation endpoints and services.",
    )
    ENABLE_MODEL_METRICS: bool = Field(
        default=True,
        description="Expose model evaluation metrics via the API and dashboard.",
    )
    ENABLE_AUDIT_LOGGING: bool = Field(
        default=True,
        description="Persist audit trail events for security-sensitive actions.",
    )
    ENABLE_RATE_LIMITING: bool = Field(
        default=True,
        description="Enable API rate limiting middleware.",
    )
    ALLOWED_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Permitted browser origins for CORS and frontend access.",
    )
    ALLOW_ALL_ORIGINS: bool = Field(
        default=False,
        description="Allow all origins by returning ['*'] for CORS configuration.",
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow cookies and authorization headers in cross-origin requests.",
    )
    CORS_ALLOW_METHODS: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        description="HTTP methods permitted for cross-origin requests.",
    )
    CORS_ALLOW_HEADERS: list[str] = Field(
        default_factory=lambda: ["Authorization", "Content-Type", "Accept"],
        description="HTTP headers permitted for cross-origin requests.",
    )

    @field_validator(
        "ALLOWED_ORIGINS",
        "CORS_ALLOW_METHODS",
        "CORS_ALLOW_HEADERS",
        "SUPPORTED_UPLOAD_EXTENSIONS",
        mode="before",
    )
    @classmethod
    def parse_comma_separated_list(cls, value: Any) -> list[str]:
        """Parse comma-separated environment values into normalized lists.

        Args:
            value: Raw environment value as a string, list, tuple, or None.

        Returns:
            list[str]: Normalized list of non-empty string values.

        Raises:
            TypeError: If the value type is not supported.
        """
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            return [item.strip() for item in stripped.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        raise TypeError(f"Expected str or list, received {type(value).__name__}.")

    @field_validator("ENVIRONMENT", mode="before")
    @classmethod
    def normalize_environment(cls, value: Any) -> str:
        """Normalize environment names to lowercase supported values.

        Args:
            value: Raw environment name from configuration input.

        Returns:
            str: Normalized environment name.

        Raises:
            TypeError: If the environment value is not a string.
        """
        if not isinstance(value, str):
            raise TypeError(f"Expected str, received {type(value).__name__}.")
        return value.strip().lower()

    @field_validator("ALLOWED_ORIGINS")
    @classmethod
    def validate_allowed_origins(cls, origins: list[str]) -> list[str]:
        """Ensure each allowed origin is a valid HTTP or HTTPS URL.

        Args:
            origins: Parsed list of allowed browser origins.

        Returns:
            list[str]: Validated and normalized origin URLs.

        Raises:
            ValueError: If any origin is not a valid HTTP or HTTPS URL.
        """
        validated_origins: list[str] = []
        for origin in origins:
            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(
                    f"Invalid origin '{origin}'. Expected format: http(s)://host[:port]."
                )
            validated_origins.append(origin.rstrip("/"))

        return validated_origins

    @field_validator("SUPPORTED_UPLOAD_EXTENSIONS")
    @classmethod
    def validate_upload_extensions(cls, extensions: list[str]) -> list[str]:
        """Ensure upload extensions are non-empty and dot-prefixed.

        Args:
            extensions: Parsed list of supported upload file extensions.

        Returns:
            list[str]: Validated lowercase file extensions.

        Raises:
            ValueError: If no extensions are configured or any extension is invalid.
        """
        if not extensions:
            raise ValueError("SUPPORTED_UPLOAD_EXTENSIONS must contain at least one extension.")

        validated_extensions: list[str] = []
        for extension in extensions:
            normalized = extension.strip().lower()
            if not normalized.startswith("."):
                raise ValueError(
                    f"Invalid upload extension '{extension}'. Extensions must start with '.'."
                )
            validated_extensions.append(normalized)

        return validated_extensions

    @field_validator("JWT_ALGORITHM")
    @classmethod
    def validate_jwt_algorithm(cls, algorithm: str) -> str:
        """Ensure the configured JWT algorithm is supported.

        Args:
            algorithm: JWT signing algorithm identifier.

        Returns:
            str: Normalized uppercase JWT algorithm.

        Raises:
            ValueError: If the algorithm is not supported.
        """
        normalized = algorithm.strip().upper()
        if normalized not in SUPPORTED_JWT_ALGORITHMS:
            supported = ", ".join(sorted(SUPPORTED_JWT_ALGORITHMS))
            raise ValueError(
                f"Unsupported JWT algorithm '{algorithm}'. Supported values: {supported}."
            )
        return normalized

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, database_url: str) -> str:
        """Ensure the database URL uses a supported PostgreSQL driver scheme.

        Args:
            database_url: SQLAlchemy database connection URL.

        Returns:
            str: Trimmed and validated database URL.

        Raises:
            ValueError: If the URL scheme or host information is invalid.
        """
        parsed = urlparse(database_url.strip())
        if parsed.scheme not in SUPPORTED_DATABASE_SCHEMES:
            supported = ", ".join(sorted(SUPPORTED_DATABASE_SCHEMES))
            raise ValueError(
                "DATABASE_URL must use a PostgreSQL-compatible scheme. "
                f"Supported values: {supported}."
            )
        if not parsed.netloc:
            raise ValueError("DATABASE_URL must include host and database information.")
        return database_url.strip()

    @field_validator("MODEL_PATH", mode="before")
    @classmethod
    def parse_model_path(cls, value: Any) -> Path:
        """Normalize the model path from string or Path input.

        Args:
            value: Raw model path value.

        Returns:
            Path: Normalized filesystem path to the model artifact.

        Raises:
            ValueError: If the model path string is empty.
            TypeError: If the value type is not supported.
        """
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("MODEL_PATH must not be empty.")
            return Path(stripped)
        raise TypeError(f"Expected str or Path, received {type(value).__name__}.")

    @field_validator("CORS_ALLOW_METHODS")
    @classmethod
    def validate_cors_methods(cls, methods: list[str]) -> list[str]:
        """Ensure CORS methods are non-empty and uppercase HTTP verbs.

        Args:
            methods: Parsed list of allowed HTTP methods.

        Returns:
            list[str]: Uppercased HTTP methods.

        Raises:
            ValueError: If no methods are configured.
        """
        if not methods:
            raise ValueError("CORS_ALLOW_METHODS must contain at least one method.")
        return [method.strip().upper() for method in methods]

    @field_validator("CORS_ALLOW_HEADERS")
    @classmethod
    def validate_cors_headers(cls, headers: list[str]) -> list[str]:
        """Ensure CORS headers are non-empty.

        Args:
            headers: Parsed list of allowed HTTP headers.

        Returns:
            list[str]: Trimmed header names.

        Raises:
            ValueError: If no headers are configured.
        """
        if not headers:
            raise ValueError("CORS_ALLOW_HEADERS must contain at least one header.")
        return [header.strip() for header in headers]

    @model_validator(mode="after")
    def validate_split_sizes(self) -> Self:
        """Ensure training, validation, and test split sizes are compatible.

        Returns:
            Settings: Validated settings instance.

        Raises:
            ValueError: If the combined split sizes exceed the available data fraction.
        """
        if self.TEST_SIZE + self.VALIDATION_SIZE >= 1.0:
            raise ValueError(
                "TEST_SIZE and VALIDATION_SIZE must sum to less than 1.0."
            )
        return self

    @model_validator(mode="after")
    def validate_cors_configuration(self) -> Self:
        """Validate CORS origin configuration based on allow-all flag.

        Returns:
            Settings: Validated settings instance.

        Raises:
            ValueError: If strict origin validation is enabled with no configured origins.
        """
        if not self.ALLOW_ALL_ORIGINS and not self.ALLOWED_ORIGINS:
            raise ValueError(
                "ALLOWED_ORIGINS must contain at least one origin when "
                "ALLOW_ALL_ORIGINS is False."
            )
        return self

    @model_validator(mode="after")
    def validate_security_constraints(self) -> Self:
        """Apply environment-sensitive security validations.

        Returns:
            Settings: Validated settings instance.

        Raises:
            ValueError: If secret length or debug logging constraints are violated.
        """
        minimum_secret_length = (
            16 if self.ENVIRONMENT in {"development", "testing"} else 32
        )
        if len(self.SECRET_KEY) < minimum_secret_length:
            raise ValueError(
                "SECRET_KEY is too short. "
                f"Minimum length is {minimum_secret_length} characters."
            )

        if self.DEBUG and self.LOG_LEVEL not in {"DEBUG", "INFO"}:
            raise ValueError("When DEBUG is enabled, LOG_LEVEL should be DEBUG or INFO.")

        if self.is_production and self.DEBUG:
            raise ValueError("DEBUG must be disabled in production.")

        if self.is_production and self.ALLOW_ALL_ORIGINS:
            raise ValueError("ALLOW_ALL_ORIGINS must be disabled in production.")

        return self

    @property
    def cors_origins(self) -> list[str]:
        """Return origins used by CORS middleware.

        Returns:
            list[str]: Allowed origins, or ['*'] when all origins are permitted.
        """
        if self.ALLOW_ALL_ORIGINS:
            return ["*"]
        return self.ALLOWED_ORIGINS

    @property
    def is_development(self) -> bool:
        """Return whether the application runs in the development environment.

        Returns:
            bool: True when ENVIRONMENT is 'development'.
        """
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        """Return whether the application runs in the production environment.

        Returns:
            bool: True when ENVIRONMENT is 'production'.
        """
        return self.ENVIRONMENT == "production"

    @property
    def log_file_path(self) -> Path:
        """Return the absolute path to the primary log file.

        Returns:
            Path: Fully qualified log file path.
        """
        return self.LOG_DIR / self.LOG_FILE_NAME

    @property
    def max_upload_size_bytes(self) -> int:
        """Return the maximum upload size in bytes.

        Returns:
            int: Maximum upload size converted from megabytes.
        """
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings.

    Returns:
        Settings: Cached settings instance.
    """
    return Settings()
