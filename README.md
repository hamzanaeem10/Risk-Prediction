# 💰 Financial Risk Prediction Platform

<div align="center">

![CI Pipeline](https://github.com/hamzanaeem10/Risk-Prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?logo=xgboost&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.19-0194E2?logo=mlflow&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B?logo=streamlit&logoColor=white)

**An end-to-end ML platform for predicting loan risk using XGBoost, Optuna hyperparameter tuning, and real-time inference via FastAPI.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Technologies](#-technologies) • [API](#-api-documentation)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏗️ **Star Schema Data Warehouse** | PostgreSQL with dimensional modeling (fact + dimension tables) |
| 🧠 **Advanced ML Pipeline** | XGBoost + RandomForest + GradientBoosting ensemble |
| 🔍 **Optuna Optimization** | 50-trial Bayesian hyperparameter tuning |
| 📊 **MLflow Tracking** | Experiment logging, model registry, artifact storage |
| ⚡ **FastAPI Backend** | High-performance async REST API |
| 🎨 **Streamlit Dashboard** | Modern dark-themed UI with gauges and charts |
| 🐳 **Fully Dockerized** | One command to run the entire stack |

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                    Streamlit (:8501)                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API LAYER                                │
│                    FastAPI (:8000)                              │
│              /predict  /health  /docs                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
┌──────────────────────┐      ┌──────────────────────────────────┐
│    MODEL ARTIFACTS   │      │          MLFLOW (:5050)          │
│  • xgboost_model.json│      │  • Experiment Tracking           │
│  • rf_model.pkl      │      │  • Model Registry                │
│  • gb_model.pkl      │      │  • Artifact Storage              │
│  • scaler.pkl        │      └──────────────────────────────────┘
└──────────────────────┘
          ▲
          │ Training
          │
┌──────────────────────────────────────────────────────────────────┐
│                      DATA WAREHOUSE                              │
│                   PostgreSQL (:5432)                            │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│  │ dim_customers │  │dim_credit_hist│  │    fact_loans     │   │
│  │ dim_geography │  │               │  │  (risk_rating)    │   │
│  └───────────────┘  └───────────────┘  └───────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Docker Compose

### One Command Start
```bash
docker-compose up --build -d
```

### First-Time Setup
```bash
# Run ETL to load data into warehouse
docker-compose exec api python src/data/etl.py

# Train model (Optuna + Ensemble)
docker-compose exec api python src/models/train.py
```

### Access Services
| Service | URL |
|---------|-----|
| 🎨 Streamlit UI | http://localhost:8501 |
| 📚 API Docs (Swagger) | http://localhost:8000/docs |
| 📊 MLflow Dashboard | http://localhost:5050 |

---

## 🛠️ Technologies

### Machine Learning
| Technology | Purpose |
|------------|---------|
| **XGBoost** | Gradient boosting for classification |
| **Optuna** | Bayesian hyperparameter optimization (50 trials) |
| **SHAP** | Model explainability & feature importance |
| **SMOTE** | Handling class imbalance |
| **Scikit-learn** | RandomForest, GradientBoosting, preprocessing |

### Data Engineering
| Technology | Purpose |
|------------|---------|
| **PostgreSQL 15** | Data warehouse with Star Schema |
| **SQLAlchemy** | ORM for database operations |
| **Pandas** | Data manipulation & ETL |

### ML Operations
| Technology | Purpose |
|------------|---------|
| **MLflow 2.19** | Experiment tracking, model registry |
| **Docker Compose** | Container orchestration |
| **Joblib** | Model serialization |

### Backend & Frontend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance REST API |
| **Streamlit** | Interactive ML dashboard |
| **Plotly** | Interactive visualizations |
| **Pydantic** | Data validation |

---

## 📡 API Documentation

### Predict Endpoint
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "age": 35,
  "gender": "Male",
  "education_level": "Bachelor's",
  "marital_status": "Married",
  "income": 75000,
  "credit_score": 720,
  "loan_amount": 25000,
  "loan_purpose": "Auto",
  "employment_status": "Employed",
  "years_at_current_job": 5,
  "payment_history": "Good",
  "debt_to_income_ratio": 0.25,
  "assets_value": 150000,
  "number_of_dependents": 2,
  "previous_defaults": 0,
  "marital_status_change": 0
}
```

**Response:**
```json
{
  "risk_class": "Low",
  "risk_probabilities": {
    "Low": 0.72,
    "Medium": 0.21,
    "High": 0.07
  },
  "confidence": 0.72
}
```

---

## 📁 Project Structure

```
financial-risk-prediction/
├── docker-compose.yml          # All services orchestration
├── docker/
│   ├── Dockerfile.api          # FastAPI container
│   └── Dockerfile.ui           # Streamlit container
├── src/
│   ├── api/main.py             # FastAPI endpoints
│   ├── data/
│   │   ├── download_data.py    # Kaggle dataset download
│   │   └── etl.py              # ETL pipeline
│   ├── models/train.py         # Optuna + Ensemble training
│   └── ui/app.py               # Streamlit dashboard
├── sql/schema.sql              # Star Schema DDL
├── models/                     # Trained model artifacts
└── requirements.txt            # Python dependencies
```

---

## 📈 Model Performance

| Metric | XGBoost | Ensemble |
|--------|---------|----------|
| Accuracy | 57% | **60%** |
| F1 (Weighted) | 47% | **54%** |

### Feature Engineering (16 Domain Features)
- `loan_to_income_ratio`, `asset_coverage_ratio`
- `high_dti_flag`, `poor_credit_flag`, `good_credit_flag`
- `job_stability`, `income_per_dependent`
- `risk_composite`, `affordability_index`
- And more...

---

## � CI/CD Pipeline

This project includes GitHub Actions for automated testing and deployment.

### Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| **CI Pipeline** | Push/PR to `main` | Lint → Test → Build Docker → Validate Model |
| **Train Pipeline** | Manual dispatch | Download data → ETL → Train → Upload artifacts |

### CI Pipeline Stages
```
🔍 Lint & Format  →  🧪 Run Tests  →  🐳 Build Images  →  🧠 Validate Model
     ↓                    ↓                  ↓                    ↓
   Ruff/Black        pytest+cov        Push to GHCR        Sanity checks
```

### Run Tests Locally
```bash
pip install pytest pytest-cov httpx
pytest tests/ -v --cov=src
```

---

## �📜 License

MIT License

---

<div align="center">
  <b>Built with ❤️ by hamza r</b>
</div>
