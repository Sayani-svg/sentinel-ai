# Sentinel AI Master Specification — Part 2

> Version: 1.0

# 1. Backend Architecture

Framework: FastAPI

Modules:
- auth
- users
- upload
- inference
- reports
- dashboard
- health
- admin

Folder:

```text
backend/
 app/
  api/
  core/
  models/
  schemas/
  services/
  database/
  middleware/
  utils/
```

---

# 2. REST APIs

## Authentication

POST /api/v1/auth/register

POST /api/v1/auth/login

POST /api/v1/auth/refresh

POST /api/v1/auth/logout

GET /api/v1/users/me

JWT Bearer Authentication.

---

## Upload

POST /api/v1/upload/logs

Supports

- CSV
- JSON

Maximum upload:
100MB

Validation:

- Required columns
- Encoding
- Missing labels

---

## Prediction

POST /api/v1/predict

Response

```json
{
  "attack":"PortScan",
  "confidence":0.97,
  "severity":"High",
  "top_features":[]
}
```

---

## Dashboard

GET /api/v1/dashboard/summary

GET /api/v1/dashboard/timeline

GET /api/v1/dashboard/top-threats

GET /api/v1/dashboard/model-metrics

---

## Reports

GET /api/v1/reports/pdf/{id}

GET /api/v1/reports/csv/{id}

POST /api/v1/reports/generate

---

# 3. Database Schema

Tables

users

incidents

predictions

uploaded_logs

reports

audit_logs

model_versions

Fields

Users

- id
- name
- email
- password_hash
- role
- created_at

Predictions

- id
- upload_id
- predicted_class
- confidence
- severity
- created_at

Reports

- id
- prediction_id
- report_path
- generated_at

---

# 4. Authentication

JWT

Password hashing:

bcrypt

Roles

- Admin
- Analyst
- Viewer

Permissions

Admin

- upload
- delete
- manage users

Analyst

- upload
- predict
- reports

Viewer

- dashboard only

---

# 5. Frontend

Stack

- React
- TypeScript
- Tailwind
- React Router
- Axios
- Recharts

Pages

- Login
- Dashboard
- Upload Logs
- Predictions
- Reports
- Analytics
- Settings

---

# 6. Dashboard Widgets

Cards

- Threats Today
- Critical Threats
- Benign Events
- Detection Rate

Charts

- Attack Timeline
- Attack Distribution
- Top Source IPs
- Confidence Histogram
- Model Metrics

Tables

Recent Incidents

Recent Uploads

---

# 7. Threat Intelligence

Severity

Critical

High

Medium

Low

Map attacks to MITRE ATT&CK techniques where feasible.

Generate recommendations based on attack type.

---

# 8. Explainability

Display

- SHAP Bar Chart
- SHAP Summary
- Top Features

Every prediction must include a human-readable explanation.

---

# 9. PDF Report

Sections

Executive Summary

Incident Details

Prediction

Confidence

Severity

Evidence

Top Features

Recommendations

MITRE Mapping

Timestamp

---

# 10. Docker

Services

frontend

backend

postgres

Volumes

models

reports

logs

Environment variables via .env

---

# 11. CI/CD

GitHub Actions

Run

- lint
- unit tests
- build backend
- build frontend

Reject failing pull requests.

---

# 12. Testing

Backend

pytest

Frontend

Vitest

Coverage target

80%

Include API integration tests.

---

# 13. Logging

Store

API logs

Prediction logs

Authentication logs

Errors

Audit trail

Structured JSON logs preferred.

---

# 14. Security

Validate uploads

Rate limiting

CORS

Security headers

Parameterized SQL

Secrets stored in environment variables only.

---

# 15. README Requirements

Include

Architecture diagram

Installation

Dataset setup

Training

Running backend

Running frontend

Docker usage

Screenshots

Future roadmap

License

---

# 16. Deployment

Frontend

Vercel

Backend

Render

Database

PostgreSQL

Models persisted under mounted volume.

---

# 17. Acceptance Criteria

✓ Upload logs

✓ Train model

✓ Load saved model

✓ Predict attacks

✓ Explain predictions

✓ Generate reports

✓ Dashboard displays analytics

✓ Authentication works

✓ Docker deployment succeeds

---

# 18. Gemini CLI Generation Instructions

Act as a senior AI engineer and full-stack architect.

Generate the repository module-by-module.

Rules:

- No placeholders.
- Fully implement every function.
- Maintain consistent architecture.
- Use dependency injection where appropriate.
- Add logging and error handling.
- Add docstrings and type hints.
- Write unit tests.
- Keep modules cohesive.
- Ensure backend, frontend, and ML integrate without breaking imports.
- After each completed module, verify internal consistency before proceeding.

# End of Part 2
