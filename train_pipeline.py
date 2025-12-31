"""
Training Pipeline for Job Role Prediction.
Handles data loading, feature engineering, model training, and evaluation.
"""
import os
import warnings
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import log_loss, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Import Shared Logic
from src.features import load_data, preprocess_features, get_feature_columns
from src.callbacks import ModelCheckpoint
from src.config import DATA_PATH, MODEL_PATH, MLFLOW_EXP_NAME, CHECKPOINT_DIR, MODEL_NAME

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# -------------------------------------------------------------------------
# 2. Pipeline Construction
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# 2. Pipeline Construction
# -------------------------------------------------------------------------

def build_pipeline(n_hash_features=1000):
    """
    Constructs the ML pipeline with preprocessing, resampling, and ensemble classifiers.
    """
    # --- Preprocessing ---
    # Pattern: HASHED FEATURE (for Skills)
    skills_transformer = HashingVectorizer(n_features=n_hash_features, alternate_sign=False, norm=None)

    # Categorical
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Composite Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('skills_hash', skills_transformer, 'skills'),
            ('cat', categorical_transformer, ['qualification', 'experience_level']),
            ('cross', categorical_transformer, ['exp_skills_cross']) # OneHot the crossed feature
        ],
        remainder='drop'
    )

    # --- Ensembles (Pattern: ENSEMBLES) ---
    # 1. XGBoost (Boosting)
    # Note: We configure it here. Callbacks are passed during fit() usually,
    # but inside VotingClassifier we can't easily pass fit_params to just one estimator.
    # WORKAROUND: We will trigger checkpointing if we were training XGB directly.
    # For the Project Requirement compliance within a VotingClassifier, we might need to
    # mock the behavior or accept that internal estimators don't easily accept unique fit_params
    # via the VotingClassifier's fit method without complex wrapper code.
    # HOWEVER, XGBClassifier allows passing callbacks in __init__ in recent versions or via fit.
    # We'll try passing it via **kwargs if supported or just demonstrate it works for standalone XGB.
    # For reliability in this specific deadline: We will train XGB independently for a few epochs
    # to DEMONSTRATE the checkpointing pattern, then include it in the Voting Ensemble.

    # -------------------------------------------------------------------------
    # Pattern: ENSEMBLES (Trade-offs Documentation)
    # -------------------------------------------------------------------------
    # We combine multiple models (XGBoost + Random Forest) using a VotingClassifier.
    #
    # Trade-offs:
    # 1. Increased Training Time: Training a Voting Ensemble effectively requires training
    #    N models instead of 1. Here, we train both an XGBoost and a Random Forest model,
    #    significantly increasing the computational cost and time per experiment check.
    # 2. Increased Design Time: Tuning an ensemble requires hyperparameter optimization
    #    for EACH underlying estimator (XGBoost params + RF params) and potentially
    #    the voting weights, making the design loop longer and more complex.
    #
    # Justification:
    # We accept these costs to reduce Variance (via Random Forest/Bagging) and Bias
    # (via XGBoost/Boosting), resulting in a more robust and stable model for production.


    xgb_clf = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        callbacks=[ModelCheckpoint(interval=10)] # Pass callback here!
    )

    # 2. Random Forest (Bagging)
    rf_clf = RandomForestClassifier(
        n_estimators=50,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # 3. Voting Classifier (Stacking/Voting)
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', xgb_clf),
            ('rf', rf_clf)
        ],
        voting='soft',
        n_jobs=-1
    )

    # Pattern: REBALANCING (RandomOverSampler)
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('rebalance', RandomOverSampler(random_state=42)),
        ('classifier', voting_clf)
    ])

    # --- Hyperparameter Tuning (Pattern: HYPERPARAMETER TUNING) ---
    # We define a search space for the underlying estimators.
    # Note: In a VotingClassifier, we target internal estimators using the 'classifier__[name]__[param]' syntax.
    param_distributions = {
        'classifier__xgb__n_estimators': [50, 100],
        'classifier__xgb__max_depth': [3, 6, 9],
        'classifier__rf__n_estimators': [50, 100],
        'classifier__rf__max_depth': [None, 10, 20]
    }

    search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_distributions,
        n_iter=5,
        cv=3,
        scoring='f1_macro',
        random_state=42,
        n_jobs=-1
    )

    return search

# -------------------------------------------------------------------------
# 3. Training & Tracking
# -------------------------------------------------------------------------

def train_job():
    """
    Main training job.
    Loads data, trains the pipeline, evaluates performance, and logs artifacts to MLflow.
    """
    # pylint: disable=too-many-locals
    mlflow.set_experiment(MLFLOW_EXP_NAME)

    with mlflow.start_run():
        # 1. Load
        df = load_data(DATA_PATH)

        # 2. Shared Feature Engineering (Pattern: FEATURE CROSS)
        df = preprocess_features(df)

        features = df[get_feature_columns()]
        target = df['job_role']

        # 3. Split
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # 4. Build Pipeline
        print("Building pipeline...")
        pipeline = build_pipeline()

        # Log Parameters
        mlflow.log_param("ensembles", "XGBoost, RandomForest, VotingClassifier")

        # 5. Train
        print("Starting Hyperparameter Search (Tuning)...")
        pipeline.fit(features_train, target_train)

        # Log Best Parameters
        best_params = pipeline.best_params_
        print(f"Best Parameters: {best_params}")
        for param, value in best_params.items():
            mlflow.log_param(f"best_{param}", value)

        # We use the best estimator for evaluation and saving
        best_model = pipeline.best_estimator_

        # 6. Evaluate
        print("Evaluating best model...")
        y_pred = best_model.predict(features_test)
        y_proba = best_model.predict_proba(features_test)

        # Metrics
        # Pattern: REBALANCING (Metric Selection)
        # Macro F1 for imbalance, Log Loss for probabilities.
        macro_f1 = f1_score(target_test, y_pred, average='macro')
        test_loss = log_loss(target_test, y_proba, labels=best_model.classes_)

        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Log Loss: {test_loss:.4f}")

        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("log_loss", test_loss)

        # 7. Save & Register
        from mlflow.models import infer_signature
        signature = infer_signature(features_test, y_pred)

        joblib.dump(best_model, MODEL_PATH)

        # Pattern: MODEL GOVERNANCE (Model Registry)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature # Explicit Schema/Signature
        )
        print(f"Model saved to {MODEL_PATH} and registered in MLflow.")

if __name__ == "__main__":
    train_job()
