# Sentinel AI — Implementation Guide
> Version 1.0

## Purpose
This document tells the coding agent **how** to build the project. It complements the Master Specification and should be followed exactly.

---

# Development Principles

1. Build production-quality code only.
2. Never leave TODOs or placeholders.
3. Every function must be fully implemented.
4. Prefer readability over cleverness.
5. Use Python type hints and docstrings.
6. Follow SOLID principles where appropriate.
7. Every module must have unit tests.
8. Every public API must validate input.
9. Every exception must be logged.
10. Keep modules cohesive.

---

# Generation Order

## Phase 1 — ML Engine

Generate in this exact order:

1. Project structure
2. Configuration
3. Logging
4. Data loading
5. Dataset validation
6. Preprocessing
7. Feature engineering
8. Model training
9. Hyperparameter tuning
10. Evaluation
11. Explainability
12. Model export
13. Inference

Validation:
- Training completes without errors.
- Metrics are written to disk.
- Saved model loads correctly.

---

## Phase 2 — Backend

Generate:

- FastAPI app
- Settings
- Database connection
- SQLAlchemy models
- Alembic configuration
- Authentication
- User management
- Upload service
- Prediction service
- Report service

Validation:
- All endpoints start successfully.
- Swagger/OpenAPI loads.
- Authentication works.

---

## Phase 3 — Frontend

Generate:

- React app
- Routing
- Authentication pages
- Dashboard
- Upload page
- Predictions page
- Reports page
- Settings

Validation:
- Frontend builds.
- API integration succeeds.
- Responsive layout verified.

---

## Phase 4 — AI Features

Implement:

- SHAP explanations
- Threat scoring
- MITRE ATT&CK mapping
- Recommendation engine
- PDF report generation

Validation:
- Explanations correspond to model output.
- PDF contains prediction details and recommendations.

---

## Phase 5 — Production

Implement:

- Docker
- Docker Compose
- CI/CD
- README
- Screenshots placeholders
- Architecture diagrams
- Deployment configuration

Validation:
- Fresh clone runs successfully.
- Docker compose starts all services.

---

# Coding Standards

## Python

- Python 3.12
- Black formatting
- Ruff compatible
- PEP8
- Use pathlib
- Avoid hard-coded paths

## React

- TypeScript
- Functional components
- Hooks
- No class components

## API

- RESTful
- JSON only
- Consistent response schema
- HTTP status codes

---

# File Responsibilities

## data_loader.py
- Load datasets
- Validate schema
- Merge files
- Save processed data

## preprocessing.py
- Missing values
- Duplicate removal
- Encoding
- Scaling
- Train/test split

## feature_engineering.py
- Variance threshold
- Correlation removal
- Mutual information
- Optional PCA

## model_training.py
- Train all required models
- Compare metrics
- Persist best model

## evaluation.py
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion matrix

## explainability.py
- SHAP explainer
- Feature importance
- Summary plots

---

# Error Handling

Always:

- Raise meaningful exceptions.
- Log stack traces.
- Never suppress exceptions silently.

---

# Logging

Every module should log:

- Start
- End
- Warnings
- Errors
- Execution time

---

# Testing

Generate:

- Unit tests
- Integration tests
- API tests
- ML pipeline smoke tests

Target:
- >=80% code coverage.

---

# Performance Goals

- Accuracy >= 90% (dataset dependent)
- Precision >= 0.90
- Recall >= 0.90
- F1 >= 0.90

Report all metrics honestly.

---

# Prompt Discipline for Gemini

When generating code:

- Generate complete files.
- Never omit imports.
- Never invent unavailable datasets.
- Keep functions under ~60 lines where practical.
- Use dependency injection where useful.
- Ensure modules import cleanly.
- Verify consistency before moving to the next module.

If context becomes too large:

1. Finish the current file.
2. Summarize completed work.
3. Continue with the next file.

---

# Completion Checklist

ML
- [ ] Data ingestion
- [ ] Preprocessing
- [ ] Training
- [ ] Evaluation
- [ ] Explainability
- [ ] Export

Backend
- [ ] Auth
- [ ] Upload
- [ ] Predict
- [ ] Reports

Frontend
- [ ] Dashboard
- [ ] Charts
- [ ] Reports

Deployment
- [ ] Docker
- [ ] CI/CD
- [ ] Documentation

Final
- [ ] End-to-end test passes
- [ ] Repository ready for GitHub

# End of Implementation Guide
