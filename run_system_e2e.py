import subprocess
import time
import requests
import os
import sys

def run_e2e():
    """
    Runs the entire MLOps system end-to-end.
    1. Runs the Prefect Workflow (CI/CD + Training)
    2. Starts the inference service
    3. Generates test traffic for monitoring
    """
    print("=== 1. Running End-to-End Workflow (Prefect) ===")
    # This covers: Linting -> Unit Tests -> Data Load -> Training -> Tuning -> Registration -> Stage Transition
    try:
        subprocess.run([sys.executable, "workflow.py"], check=True)
        print("[SUCCESS] Workflow completed successfully.")
    except subprocess.CalledProcessError:
        print("[ERROR] Workflow failed. Check logs.")
        return

    print("\n=== 2. Starting Inference Service (FastAPI) ===")
    # Run uvicorn in background
    # Note: On Windows, use 'start' or just run it and stop later.
    # We will use subprocess.Popen
    env = os.environ.copy()
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "inference_service:app", "--host", "0.0.0.0", "--port", "8000"],
        env=env
    )
    
    time.sleep(5) # Wait for startup
    
    print("\n=== 3. Generating Evidence (Prediction Requests) ===")
    endpoint = "http://localhost:8000/predict"
    test_cases = [
        {"skills": "Python, ML, Docker", "qualification": "Masters", "experience_level": "Senior"},
        {"skills": "Java, Spring", "qualification": "B.Tech", "experience_level": "Mid"},
        {"skills": "React, CSS", "qualification": "B.Tech", "experience_level": "Junior"},
        {"skills": "Excel", "qualification": "Intern", "experience_level": "Intern"}, # Likely to trigger fallback
        {"skills": "Kubernetes, Go, AWS", "qualification": "PhD", "experience_level": "Executive"},
    ]
    
    for i, payload in enumerate(test_cases):
        try:
            res = requests.post(endpoint, json=payload)
            print(f"Request {i+1}: Result -> {res.json()['predicted_role']} (Status: {res.json().get('status', 'Success')})")
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
        time.sleep(1)

    print("\n=== 4. Verifying Monitoring Dashboard (Prometheus) ===")
    try:
        metrics_res = requests.get("http://localhost:8000/metrics")
        if metrics_res.status_code == 200:
            print("✅ Prometheus metrics endpoint is LIVE.")
            print("✅ Prometheus metrics endpoint is LIVE.")
            print("Sample Metrics (Full Output):")
            print(metrics_res.text)
    except Exception as e:
        print(f"Metrics check failed: {e}")

    print("\n=== Evidence Capture Complete ===")
    print("You can now view:")
    print("1. MLflow UI: run 'mlflow ui' and check 'JobRole_Prediction_Production'")
    print("2. Monitoring: check http://localhost:8000/metrics")
    print("3. Logs: check workflow outputs for CI/CD evidence")
    
    print("\nPress Ctrl+C to stop the API service...")
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down service...")
        process.terminate()

if __name__ == "__main__":
    run_e2e()
