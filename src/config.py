import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'candidate_job_role_dataset.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'voting_clf.joblib')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# MLflow
MLFLOW_EXP_NAME = 'JobRole_Prediction_Production'
MODEL_NAME = "JobRolePredictor"

# Serving
FALLBACK_THRESHOLD = 0.6
DRIFT_THRESHOLD = 0.5
WINDOW_SIZE = 50

# Linting
LINT_FILES = ["inference_service.py", "train_pipeline.py", "workflow.py", "src/", "tests/"]
LINT_THRESHOLD = 7.0
