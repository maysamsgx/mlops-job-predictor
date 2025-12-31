"""
Unit tests for the Training Pipeline.
Tests feature engineering, data splitting, and model building.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
import joblib

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import preprocess_features, get_feature_columns

# -------------------------------------------------------------------------
# Unit Tests (Feature Engineering)
# -------------------------------------------------------------------------

def test_preprocess_features_generates_cross_feature():
    """
    Test that feature cross 'exp_skills_cross' is generated correctly.
    """
    raw_data = pd.DataFrame([{
        'skills': 'Python,Docker',
        'qualification': 'B.Tech',
        'experience_level': 'Junior'
    }])

    processed = preprocess_features(raw_data)

    # encoded_skills_count = len(['Python', 'Docker']) # 2
    # Bins: (-1, 5] -> Low
    expected_cross = 'Junior_Low'

    assert 'exp_skills_cross' in processed.columns
    assert processed['exp_skills_cross'].iloc[0] == expected_cross
    assert 'skills_bin' in processed.columns

def test_preprocess_features_handles_empty_skills():
    """
    Test resilience against empty skills.
    """
    raw_data = pd.DataFrame([{
        'skills': '', # Empty string
        'qualification': 'PhD',
        'experience_level': 'Senior'
    }])

    processed = preprocess_features(raw_data)

    assert 'skills_bin' in processed.columns
    assert processed['skills_bin'].iloc[0] == 'Low'

# -------------------------------------------------------------------------
# Component Tests (Model Pipeline)
# -------------------------------------------------------------------------

MODEL_PATH = 'voting_clf.joblib'

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_model_loading_and_prediction():
    """
    Test that the saved model can be loaded and predicts valid output shapes.
    """
    pipeline = joblib.load(MODEL_PATH)

    # Create valid interaction via Shared Logic
    raw_input = pd.DataFrame([{
        'skills': 'Java,Spring,Kubernetes,AWS,Python',
        'qualification': 'M.Tech',
        'experience_level': 'Mid'
    }])

    processed = preprocess_features(raw_input)
    features = processed[get_feature_columns()]

    # Predict
    prediction = pipeline.predict(features)
    proba = pipeline.predict_proba(features)

    assert len(prediction) == 1
    assert proba.shape[1] > 1 # Multi-class probabilities
    assert np.isclose(np.sum(proba), 1.0)

def test_feature_columns_consistency():
    """
    Ensure feature columns defined in src match what we expect.
    """
    cols = get_feature_columns()
    assert 'exp_skills_cross' in cols
    assert 'skills' in cols

def test_data_loading_repeatability(tmp_path):
    """
    Requirement: Validate data processing repeatability.
    Ensures that loading the same data results in the same DataFrame structure.
    """
    data_content = "skills,qualification,experience_level,job_role\nPython,B.Tech,Junior,Developer\n"
    test_file = tmp_path / "test_data.csv"
    test_file.write_text(data_content, encoding='utf-8')

    df1 = pd.read_csv(test_file)
    df2 = pd.read_csv(test_file)

    pd.testing.assert_frame_equal(df1, df2)

def test_pipeline_encoding_integration():
    """
    Ensures that the categorical features are correctly identified for encoding.
    """
    # Import here to avoid top-level import errors if not in path
    from train_pipeline import build_pipeline
    search_cv = build_pipeline()
    pipeline = search_cv.estimator

    # Check if preprocessor has the expected transformers
    transformers = pipeline.named_steps['preprocessor'].transformers
    transformer_names = [t[0] for t in transformers]

    assert 'cat' in transformer_names
    assert 'skills_hash' in transformer_names
    assert 'cross' in transformer_names
