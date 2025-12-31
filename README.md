# Job Role Prediction System (MLOps Level 2)

**Team Group 3**  
Target: MLOps Level 2 Maturity (CI/CD Pipeline Automation)

## 🚀 Project Overview
This system is an automated, resilient Machine Learning pipeline designed to predict job roles from high-cardinality candidate data. It features:
- **Automated Training Pipeline** (Prefect + XGBoost/RandomForest)
- **Model Governance** (MLflow Registry)
- **Resilient Serving** (FastAPI + Algorithmic Fallback)
- **Infrastructure** (Docker + Kubernetes Rolling Updates)

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.9+
- Docker (Optional, for containerization)

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd CHECK

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the System

**Option A: Full End-to-End Demo (Recommended)**
Runs the entire pipeline (Training -> Serve -> Predict -> Monitor) in one script.
```bash
python run_system_e2e.py
```

**Option B: Manual Pipeline Trigger**
```bash
python workflow.py
```

**Option C: Start Inference Service**
```bash
uvicorn inference_service:app --reload
```

## 📊 Key Artifacts
- **Training Pipeline**: `train_pipeline.py`
- **Inference Service**: `inference_service.py`
- **Orchestration**: `workflow.py`
- **Configuration**: `src/config.py`
- **CI/CD**: `.gitlab-ci.yml`

## 🧪 Testing
```bash
pytest tests/
```

## 📈 Monitoring
- **MLflow UI**: `http://localhost:5000`
- **Prometheus Metrics**: `http://localhost:8000/metrics`
