# Sentinel AI Development Mandates & Workspace Architecture (GEMINI.md)

This document contains team-shared architecture guidelines, coding conventions, workflows, and AI prompt discipline instructions for Sentinel AI. It is a foundational mandate that takes absolute precedence for all development and code generation within this repository.

---

## Table of Contents
1. [Core Architecture & Repository Structure](#1-core-architecture--repository-structure)
2. [Team-Shared Coding Standards & Conventions](#2-team-shared-coding-standards--conventions)
3. [Technology Stack](#3-technology-stack)
4. [Phase-by-Phase Development Workflow](#4-phase-by-phase-development-workflow)
5. [Database Schema & API Specifications](#5-database-schema--api-specifications)
6. [Testing & Quality Gates](#6-testing--quality-gates)
7. [AI Generation Protocols & Prompt Discipline](#7-ai-generation-protocols--prompt-discipline)

---

## 1. Core Architecture & Repository Structure

Sentinel AI is organized as a monorepo consisting of three main functional tiers (Machine Learning, Backend API, and Frontend SPA), along with auxiliary infrastructure, tests, and documentation.

```text
sentinel-ai/
├── backend/            # FastAPI Application and Alembic Migrations
│   ├── app/            # Main FastAPI source folder
│   │   ├── api/        # REST endpoints (v1 routes: auth, upload, predict, dashboard, etc.)
│   │   ├── core/       # Configurations, security dependencies, and settings
│   │   ├── database/   # Base model and session management
│   │   ├── middleware/ # CORS, Rate Limiting, Security Headers
│   │   ├── models/     # SQLAlchemy database models
│   │   ├── schemas/    # Pydantic schemas for request/response serialization
│   │   ├── services/   # Business logic (auth, user, upload, prediction, reports)
│   │   └── utils/      # Threat intelligence, recommendations, logging helpers
│   └── tests/          # Pytest unit and integration tests
├── frontend/           # React + TypeScript Single Page Application (Vite)
│   ├── public/         # Static assets
│   └── src/            # Client-side source code
│       ├── components/ # Reusable UI components (charts, dashboard widgets, layout)
│       ├── hooks/      # Custom React Hooks (e.g., useAuth)
│       ├── pages/      # Route level components (Dashboard, Analytics, Predictions, etc.)
│       ├── services/   # API clients (axios wrappers for backend endpoints)
│       ├── types/      # TypeScript declarations and interfaces
│       └── utils/      # Constants and formatting utilities
├── ml/                 # Machine Learning pipeline and model assets
│   ├── dataset/        # Benchmark training sets (CICIDS2017 raw/processed data)
│   ├── models/         # Saved model files, encoders, and scalers
│   ├── outputs/        # Performance logs, confusion matrices, and SHAP plots
│   ├── src/            # ML engine source (data loader, preprocessing, training, SHAP)
│   └── tests/          # Pipeline smoke tests and unit tests for preprocessing
├── docker/             # Containerization assets (Compose and Dockerfiles)
├── docs/               # System documentation and architecture diagrams
└── tests/              # Project-level Integration and End-to-End tests
```

---

## 2. Team-Shared Coding Standards & Conventions

To maintain a clean, readable, and highly maintainable codebase, all developers and AI agents must strictly follow these coding standards.

### General Engineering Principles
- **No Placeholders / No TODOs:** Every file generated must be fully implemented, with all imports specified, proper return types, error handling, and logical completeness.
- **Readability Over Cleverness:** Keep code clear, explicit, and easy to trace. Avoid complex metaprogramming, reflection, or prototype manipulation unless absolutely necessary.
- **Composition over Inheritance:** Prefer modular composition, dependency injection, and wrapper/delegator patterns to long or nested inheritance hierarchies.

### Backend & ML (Python 3.12)
- **Type Hints:** Explicit Python type hints are **mandatory** on all function signatures, variable assignments where types are ambiguous, and class attributes.
- **Docstrings:** Standardized docstrings (Sphinx or Google style) must accompany every public class, method, and helper function, outlining parameters, return types, and raised exceptions.
- **Logging over Printing:** Never use the `print()` statement. Use Python’s standard `logging` module to output structured logs (INFO, WARNING, ERROR, DEBUG) for state changes, timings, and exceptions.
- **Path Management:** Always use standard `pathlib.Path` objects. Never hardcode absolute file paths or use string-based concatenation for OS filesystems.
- **Exceptions:** Never silently suppress exceptions. Always log the full traceback, and raise precise, context-rich custom exceptions when system boundaries are violated.

### Frontend (React & TypeScript)
- **Functional Components:** All components must be written as modern functional components utilizing React Hooks. Avoid legacy Class-based components.
- **Type Safety:** Maintain 100% strict TypeScript types. Avoid the use of `any` casts or bypassing the type checker via compiler suppressions.
- **Styling:** Use standard utility-first Tailwind CSS. Keep layout margins, padding, and colors aligned with the master theme guidelines.
- **Components Cohesion:** Keep components small, modular, and single-purpose. Separate presentation from data-fetching hooks.

---

## 3. Technology Stack

- **Machine Learning:** Python 3.12, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna (hyperparameter search), SHAP (explainability), pandas, numpy.
- **Backend Framework:** FastAPI (Asynchronous Python ASGI), SQLAlchemy ORM, Alembic (database migration manager), JWT authentication.
- **Database:** PostgreSQL (production database) / SQLite (development/testing where specified).
- **Frontend Framework:** React 18+, TypeScript, Tailwind CSS, Recharts (for dynamic SOC data visualizations), Axios (for client API requests), Vite (as bundler/dev server).
- **Deployment:** Docker, Docker Compose, GitHub Actions (CI/CD pipeline), Render (Backend hosting), Vercel (Frontend hosting).

---

## 4. Phase-by-Phase Development Workflow

The implementation of Sentinel AI proceeds sequentially through these five logical phases.

### Phase 1: Machine Learning Engine
Generate pipeline modules in this exact order, validating each before moving forward:
1. `config.py` (Centralize model settings, hyperparameter scopes, and filesystem paths)
2. `logger.py` (Structured logging utility)
3. `data_loader.py` (Dataset ingestion, basic shape verification, schema alignment)
4. `dataset_validation.py` (Integrity checks, target column validation, null rate analysis)
5. `preprocessing.py` (De-duplication, missing value imputation, label encoding, scaling, train/test split)
6. `feature_engineering.py` (Correlation filtering, constant column removal, VarianceThreshold, Mutual Information)
7. `model_training.py` (Train Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost with uniform datasets)
8. `hyperparameter_tuning.py` (Optimize parameters with Optuna over at least 50 trials)
9. `evaluation.py` (Export metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
10. `explainability.py` (Compute SHAP values, feature importance, generate summary & waterfall plots)
11. `model_export.py` (Save `best_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, and `metadata.json`)
12. `inference.py` (Load assets, receive raw log rows, perform inference, and return JSON structure)

### Phase 2: Backend REST Services
1. Set up FastAPI Core (`main.py`, `config.py`, database setup via SQLAlchemy and Alembic).
2. Establish API Route routers inside `/api/v1/`:
   - `/auth` (JWT generation, password hashing via bcrypt, refresh tokens).
   - `/users` (Registration, login, `/me` profile extraction, roles checks).
   - `/upload` (Accept CSV/JSON logs up to 100MB, perform basic structural validation).
   - `/predict` (Trigger the ML inference pipeline on uploaded logs, persist outcomes to database).
   - `/reports` (On-demand PDF/CSV generation including evidence summary, recommendations, and timestamp).
   - `/dashboard` (Aggregates for recent uploads, threats timeline, distribution charts, and source IPs).
   - `/health` (Database and pipeline liveness checks).
   - `/admin` (User roles administration and deletion of logs/incidents).

### Phase 3: Frontend Client Application
1. Initialize React + Vite + TS structure, configure Axios interceptors for handling JWT authorization headers.
2. Formulate App Routes using React Router:
   - `/login` (User credentials forms, session management via `useAuth`).
   - `/dashboard` (SOC visual panels: key metrics cards, attack distribution charts, recent incidents table).
   - `/upload-logs` (Drag-and-drop log submission, tracking progress, status messages).
   - `/predictions` (Interactive table of run classifications, filterable by severity and class).
   - `/reports` (Report catalog, download PDF triggers, system export parameters).
   - `/analytics` (Model metrics, SHAP summary visuals, top features list).
   - `/settings` (User details and profile attributes).

### Phase 4: Threat Intelligence & AI Integration
1. **MITRE ATT&CK Mapping:** Maintain a static JSON/dict utility mapping detected attack classes (e.g., DoS Hulk, PortScan) to official MITRE techniques and tactics.
2. **Recommendation Engine:** Dynamically append actionable security remediation guidelines to predictions based on the classified threat type and severity score.
3. **SHAP Renderers:** Implement frontend UI components capable of rendering SHAP summary and feature importance plots using SVG or Recharts.

### Phase 5: Production Deployment & CI/CD
1. Compose Docker configuration (`Dockerfile.backend`, `Dockerfile.frontend`, and `docker-compose.yml`) ensuring separate volumes are set up for models, reports, and persistent database folders.
2. Script GitHub Actions Workflow (`ci.yml`) to run linting, unit tests, and build checks automatically on every pull request.

---

## 5. Database Schema & API Specifications

### Database Schema Guidelines
All models must extend from the common SQLAlchemy declarative base class.
- **`users`:** `id` (PK), `name`, `email` (Unique), `password_hash`, `role` (Admin, Analyst, Viewer), `created_at`.
- **`uploaded_logs`:** `id` (PK), `filename`, `file_path`, `upload_status`, `uploaded_by` (FK to users), `created_at`.
- **`predictions`:** `id` (PK), `upload_id` (FK to uploaded_logs), `attack_type` (e.g., Benign, Bot, DoS), `confidence`, `severity` (Critical, High, Medium, Low), `created_at`.
- **`reports`:** `id` (PK), `prediction_id` (FK to predictions), `report_path`, `generated_at`.
- **`audit_logs`:** `id` (PK), `user_id` (FK to users), `action_performed`, `timestamp`, `ip_address`.
- **`model_versions`:** `id` (PK), `version_tag`, `model_type`, `accuracy`, `precision`, `recall`, `f1_score`, `is_active`, `created_at`.

### API Design Principles
- Ensure API routes strictly return JSON structures matching Pydantic response schemas.
- Incorporate proper HTTP status codes (`200 OK`, `201 Created`, `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `500 Internal Server Error`).
- Integrate CORS, API Rate Limiting, and standardized Security Headers in FastAPI middlewares.

---

## 6. Testing & Quality Gates

Quality and reliability are verified through rigorous, automated quality gates. **A change is never considered complete until it has passed all tests and code style verifications.**

- **Coverage Target:** Minimum **80% code coverage** across both frontend and backend modules.
- **Python Verification:** Run `pytest` for unit and integration testing. Ensure code formatting is verified with `black --check` or `ruff check`.
- **Frontend Verification:** Run `vitest` for React tests and verify TypeScript compiled state without warnings or errors (`npm run build`).
- **No Hacks Rule:** Never suppress linter warnings or TypeScript type warnings via bypass flags like `@ts-ignore` or inline `noqa` comments unless explicitly documented and approved.

---

## 7. AI Generation Protocols & Prompt Discipline

When operating under Gemini CLI, developers and AI agents must follow this discipline:

- **Incremental Integration:** Build the repository module-by-module. After generating any backend service, model, or component, run build/lint verification scripts before moving on.
- **Complete Outputs Only:** Never emit partial, truncated, or stubbed files. Avoid using comment lines like `// ... rest of code unchanged` or `# TODO: implement`. Complete every single function block with robust logic.
- **Deterministic Training:** Always pass fixed random seeds (`random_state=42` or similar) to ensure ML training pipeline consistency across multiple runs.
- **Empirical Validation:** Prior to applying any bug fixes, construct a failing test case or reproduction script to verify the defect, and confirm that the fix resolves the failure without causing regressions.

---
**This document serves as the repository's single source of truth for engineering instructions. Adhere to it precisely.**
