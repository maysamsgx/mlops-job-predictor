"""
Stateless Inference Service for Job Role Prediction.
Implements FastAPI endpoint, Prometheus monitoring, and Drift Detection.
"""
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, generate_latest
from fastapi.responses import Response

# Import Shared Logic
import uvicorn
from src.features import preprocess_features, get_feature_columns
from src.config import MODEL_PATH, FALLBACK_THRESHOLD, DRIFT_THRESHOLD, WINDOW_SIZE

# Metrics for CME (Pattern: CONTINUOUS EVALUATION)
PREDICTION_COUNTER = Counter('prediction_count', 'Total predictions served')
FALLBACK_COUNTER = Counter('fallback_count', 'Total fallback triggers')
CONFIDENCE_GAUGE = Gauge('model_confidence_avg', 'Rolling average of model confidence')

# Global State for Monitoring
confidence_history = []

# --- 1. Service Setup (Pattern: STATELESS SERVING) ---
# We implement a REST endpoint for real-time, low-latency predictions.
#
# Trade-off Discussion: BATCH SERVING
# Alternative: Batch Serving (e.g., running predictions on a large CSV overnight).
# When appropriate:
# - High-throughput, non-latency-sensitive workloads (e.g., scoring millions of candidates/day).
# - Asynchronous processing where immediate feedback is not required.
# - Cost optimization (using spot instances).
#
# Why Stateless REST here?
# - Supports autoscaling (horizontal scaling of stateless containers).
# - Language-neutral interface (any client can call HTTP JSON).
# - Immediate feedback for user-facing applications.

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        print("[OK] Model loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load model: {e}")
    yield
    print("Shutting down service...")

app = FastAPI(title="Job Role Prediction Service", lifespan=lifespan)

# --- 2. Schemas ---

class CandidateProfile(BaseModel):
    skills: str  # e.g. "Python, Docker, SQL"
    qualification: str # e.g. "B.Tech"
    experience_level: str # e.g. "Senior"

class PredictionResponse(BaseModel):
    predicted_role: str
    confidence: float
    status: str # "Success" or "Fallback"

# --- 3. Helper Functions ---

def update_monitoring(confidence):
    """
    Pattern: C.M.E (Continuous Model Evaluation)
    Updates rolling metrics and Checks for Drift.
    """
    # global confidence_history is not needed for list mutation (.append/.pop)
    confidence_history.append(confidence)
    if len(confidence_history) > WINDOW_SIZE:
        confidence_history.pop(0)

    avg_conf = np.mean(confidence_history)
    CONFIDENCE_GAUGE.set(avg_conf)

    # Simple Drift Check based on Confidence Drop
    if len(confidence_history) == WINDOW_SIZE and avg_conf < DRIFT_THRESHOLD:
        print(f"[ALERT] MODEL PERFORMANCE DEGRADATION DETECTED!")
        print(f"   Avg Confidence ({avg_conf:.2f}) < Threshold ({DRIFT_THRESHOLD})")
        print(f"   Action: Triggering Retraining Pipeline... (Simulated)")

def validate_input_statistics(df):
    """
    Pattern: MONITORING (Statistical Checks / Feature Validation)
    Performs 'Great Expectations'-style validation on input data.
    """
    # Check 1: Missing Values
    if df.isnull().any().any():
        raise ValueError("Input data contains null values")

    # Check 2: Statistical/Domain Checks
    # Example: Experience Level Validation (Domain Constraint)
    valid_levels = {'Junior', 'Mid', 'Senior', 'Executive', 'Intern'} # Simplified valid set
    if not df['experience_level'].iloc[0] in valid_levels:
        # We don't crash, but we log this as a 'Shift' or 'Anomaly'
        print(f"[WARNING] Unseen Experience Level: {df['experience_level'].iloc[0]}")
        # In a real system, we'd increment a 'feature_skew_counter' metric here

    # Check 3: Essential Feature Presence
    if len(df['skills'].iloc[0]) < 2:
        print(f"[WARNING] Potentially malformed 'skills' input (too short).")

# --- 4. Main Endpoint (Pattern: ALGORITHMIC FALLBACK) ---

@app.post("/predict", response_model=PredictionResponse)
def predict(profile: CandidateProfile):
    # global model
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Preprocess Input
    try:
        # Create raw dataframe
        input_data = pd.DataFrame([{
            'skills': profile.skills,
            'qualification': profile.qualification,
            'experience_level': profile.experience_level
        }])

        # Pattern: MONITORING (Feature Validation)
        validate_input_statistics(input_data)

        # Apply Shared Feature Engineering (Removes Training-Serving Skew!)
        processed_data = preprocess_features(input_data)

        # Select columns expected by 'preprocessor'
        features = processed_data[get_feature_columns()]

        # 2. Inference
        target_classes = model.classes_
        proba = model.predict_proba(features)[0] # Array of probabilities

        max_prob_idx = np.argmax(proba)
        max_conf = proba[max_prob_idx]
        predicted_role = target_classes[max_prob_idx]

        # Update Monitoring
        PREDICTION_COUNTER.inc()
        update_monitoring(max_conf)

        # 3. Fallback Logic (Pattern: FALLBACK)
        # Strategy:
        # A. Per-Prediction Fallback:
        #    If confidence < FALLBACK_THRESHOLD, we return a hard-coded "Generalist_Candidate_Review_Required" response.
        #    This ensures we don't serve low-confidence garbage to the user.
        #
        # B. System-Level Fallback (Strategy Presentation):
        #    If 'update_monitoring' detects drift (avg confidence < DRIFT_THRESHOLD), the system raises an ALERT.
        #    In a widespread outage, we would:
        #    1. Rollback: Automatically redeploy the previous 'voting_clf.joblib' version.
        #    2. Circuit Breaker: If rollback fails, switch to a purely rule-based heuristic (e.g. keyword matching)
        #       serving from a separate, simpler service until retraining completes.
        if max_conf < FALLBACK_THRESHOLD:
            FALLBACK_COUNTER.inc()
            return {
                "predicted_role": "Generalist_Candidate_Review_Required",
                "confidence": float(max_conf),
                "status": f"Fallback_Triggered (Conf < {FALLBACK_THRESHOLD})"
            }

        return {
            "predicted_role": predicted_role,
            "confidence": float(max_conf),
            "status": "Success"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
