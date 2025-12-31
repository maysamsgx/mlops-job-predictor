"""
MLOps Orchestration Workflow using Prefect.
Handles Linting, Unit Testing, Training, and Registry operations.
"""
import os
import subprocess
import sys
import joblib
import mlflow
from prefect import flow, task

# Constants
from src.config import MODEL_PATH, MLFLOW_EXP_NAME, LINT_FILES, LINT_THRESHOLD

@task(name="Load Data", retries=2)
def load_data_step():
    """
    Simulates data validation and loading checking.
    """
    print("Checking if data file exists...")
    if not os.path.exists('candidate_job_role_dataset.csv'):
        raise FileNotFoundError("Dataset not found!")
    print("Data found.")

@task(name="Run Training Pipeline")
def run_training_step():
    """
    Executes the training script.
    We run it as a subprocess to ensure environment isolation or just call the function.
    Calling the function is better if imports are clean.
    """
    from train_pipeline import train_job
    print("Starting Training Job...")
    train_job()
    print("Training Job Completed.")

@task(name="Validate Model")
def validate_model_step():
    """
    Checks if model artifact exists and meets basic criteria.
    """
    print("Validating Model Artifact...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model artifact not found after training!")

    model = joblib.load(MODEL_PATH)
    print(f"Model loaded: {type(model)}")
    print("Validation passed.")

@task(name="Deploy to Staging")
def deploy_step():
    """
    Pattern: MODEL GOVERNANCE (Stage Management)
    Transitions the latest model version to 'Staging'.
    """
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    model_name = "JobRolePredictor"

    print(f"Fetching latest versions for model: {model_name}...")
    latest_versions = client.get_latest_versions(model_name, stages=["None"])

    if latest_versions:
        latest_version = latest_versions[0].version
        print(f"Transitioning version {latest_version} to 'Staging'...")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging"
        )
        print(f"[OK] Version {latest_version} successfully moved to Staging.")
    else:
        print("[WARN] No new model versions found to transition.")


def _run_flake8():
    """Runs Flake8 to enforce PEP8 standards."""
    print("1. Running flake8 (Basic Standards)...")
    cmd = [sys.executable, "-m", "flake8"] + LINT_FILES + ["--count", "--select=E9,F63,F7,F82", "--show-source"]
    subprocess.run(cmd, check=True)

def _run_pylint():
    """Runs Pylint to enforce architectural standards."""
    print("2. Running pylint (Architectural Consistency)...")
    cmd = [sys.executable, "-m", "pylint"] + LINT_FILES + [f"--fail-under={LINT_THRESHOLD}", "--ignore=.venv"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"[ERROR] Pylint violations found:\n{result.stdout}")
        raise RuntimeError("Build failed due to Pylint architectural violations.")
    print(f"Pylint Checks Passed (Score >= {LINT_THRESHOLD}).")

def _run_safety():
    """Runs Safety to check for dependency vulnerabilities."""
    print("3. Running safety (Dependency Vulnerabilities)...")
    result = subprocess.run([sys.executable, "-m", "safety", "check"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"[WARNING] Dependency Vulnerabilities Found:\n{result.stdout}")
    else:
        print("[SUCCESS] Static Analysis Passed.")

@task(name="Run Static Analysis (Linting & Security)")
def run_linting_step():
    """
    Pattern: STATIC ANALYSIS
    Enforces Coding Standards (pylint/flake8) and Dependency Security (safety).
    """
    _run_flake8()
    _run_pylint()
    _run_safety()

@task(name="Run Unit Tests")
def run_unit_tests_step():
    """
    Requirement: Automated Unit Testing
    Ensures feature engineering and hashing logic works correctly.
    """
    print("Running Unit Tests (pytest)...")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_hashing.py", "tests/test_pipeline.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAIL] Unit Tests Failed:\n{result.stdout}")
        raise RuntimeError("Build failed due to failing unit tests.")
    print("[OK] Unit Tests Passed.")

@flow(name="MLOps_Training_Workflow")
def mlops_workflow():
    """
    Orchestrates the End-to-End MLOps Pipeline.

    Justification for using Prefect:
    1. Dynamic Workflows: Enables intelligent handling of data changes (e.g., skipping training if data hasn't changed).
    2. Fail Fast: If 'Load Data' or 'Validate Input' fails, the pipeline halts immediately, preventing resource
       wastage on a broken training run. This is critical for complex retraining loops.
    """
    # Fail Fast: Run Checks First
    run_linting_step()
    run_unit_tests_step()

    # Core Pipeline
    load_data_step()
    run_training_step()
    validate_model_step()
    deploy_step()

if __name__ == "__main__":
    mlops_workflow()
