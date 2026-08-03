# Sentinel AI Master Specification — Part 1

> Version: 1.0

## 1. Project Overview

**Project Name:** Sentinel AI

**Objective:** Build an AI-powered Security Operations Center (SOC) platform capable of ingesting security logs, detecting malicious activity using machine learning, explaining predictions, and presenting results in a professional dashboard.

### Goals

- Detect cyber attacks from structured logs
- Classify attack categories
- Explain predictions using SHAP
- Provide REST APIs
- Support deployment using Docker
- Maintain production-quality code

---

# 2. Tech Stack

## Machine Learning
- Python 3.12
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Optuna
- SHAP
- pandas
- numpy

## Backend
- FastAPI
- SQLAlchemy
- Alembic
- JWT Authentication

## Database
- PostgreSQL

## Frontend
- React
- TypeScript
- Tailwind CSS
- Recharts

## Deployment
- Docker
- Docker Compose
- GitHub Actions
- Render

---

# 3. Datasets

Primary:
- CICIDS2017

Optional:
- CSE-CIC-IDS2018
- UNSW-NB15

Target column:
Label

Classes include:
- Benign
- DoS Hulk
- GoldenEye
- PortScan
- DDoS
- Bot
- Web Attack
- Brute Force
- SQL Injection
- XSS

---

# 4. Repository Structure

```text
sentinel-ai/
 backend/
 frontend/
 ml/
 docs/
 docker/
 tests/
```

ML Structure

```text
ml/
 dataset/
 models/
 outputs/
 src/
 tests/
```

---

# 5. Coding Standards

- PEP8
- Type hints mandatory
- Docstrings on public methods
- Logging instead of print()
- No global mutable state
- Unit-testable architecture

---

# 6. Machine Learning Pipeline

1. Load dataset
2. Validate schema
3. Remove duplicates
4. Handle missing values
5. Encode labels
6. Scale features
7. Train/test split
8. Feature selection
9. Train multiple models
10. Hyperparameter tuning
11. Evaluate
12. Explain predictions
13. Export best model

---

# 7. Feature Engineering

Implement:

- Missing value handling
- Duplicate removal
- Constant feature removal
- Correlation filtering
- VarianceThreshold
- Mutual Information ranking
- Optional PCA

---

# 8. Models To Train

Mandatory:

- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost

Optional:

- Isolation Forest
- Logistic Regression

Train every model using identical preprocessing.

---

# 9. Hyperparameter Tuning

Use Optuna.

Optimize:

- n_estimators
- max_depth
- learning_rate
- min_samples_leaf
- subsample
- colsample_bytree

Trials:
Minimum 50.

---

# 10. Evaluation

Generate:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion Matrix
- Classification Report

Save all outputs.

---

# 11. Explainability

Use SHAP.

Produce:

- Summary plot
- Feature importance
- Waterfall plot
- Force plot (optional)

Generate explanations for every prediction.

---

# 12. Model Export

Export:

```
best_model.pkl
scaler.pkl
label_encoder.pkl
metadata.json
```

Metadata should include:
- dataset
- timestamp
- model
- metrics
- feature names

---

# 13. Logging

Use Python logging.

Levels:
- INFO
- WARNING
- ERROR

Every pipeline stage should emit structured logs.

---

# 14. Error Handling

Handle:

- Missing dataset
- Invalid schema
- Empty dataframe
- Failed model training
- Invalid labels
- Serialization failures

Never silently ignore exceptions.

---

# 15. Performance Targets

Desired goals on benchmark datasets:

- Accuracy >= 90%
- Precision >= 0.90
- Recall >= 0.90
- F1 >= 0.90

Do not hardcode results.

---

# 16. Deliverables

At completion of Part 1:

- Working ML pipeline
- Saved model
- Evaluation report
- SHAP explanations
- Metrics JSON
- Training logs

---

# 17. Gemini CLI Generation Protocol

You are acting as a Senior Machine Learning Engineer.

Rules:

1. Generate production-quality code only.
2. Never generate placeholders.
3. Never generate TODO comments.
4. Complete every function.
5. Use type hints.
6. Use logging.
7. Write modular code.
8. Prefer composition over inheritance.
9. Keep files under 500 lines where practical.
10. Generate unit tests.
11. Follow repository structure exactly.
12. Stop after completing one module if interactive mode is requested.
13. Ensure all imports resolve.
14. Code must run on Python 3.12.
15. Use deterministic random seeds where appropriate.

# End of Part 1
