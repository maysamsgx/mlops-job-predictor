import pandas as pd
import numpy as np

def load_data(path):
    """
    Standardized data loading.
    """
    return pd.read_csv(path)

def preprocess_features(df):
    """
    Applies shared feature engineering logic (Feature Cross, Binning).
    Usage:
    - Training: Call this on the raw dataframe before split.
    - Inference: Call this on the single-row dataframe before passing to pipeline.
    """
    df = df.copy()
    
    # Ensure skills_count exists (Handle missing if necessary, though assumed present in raw or created steps)
    if 'skills_list' not in df.columns:
        # Check if 'skills' column exists and is not null
        if 'skills' in df.columns:
            # Handle potential NaN in skills
            df['skills'] = df['skills'].fillna('')
            df['skills_list'] = df['skills'].astype(str).str.split(',')
            df['skills_count'] = df['skills_list'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        else:
            # Fallback if somehow skills is missing entirely (shouldn't happen in this dataset)
            df['skills_count'] = 0

    # Bin skills count (Low, Med, High)
    df['skills_bin'] = pd.cut(df['skills_count'], bins=[-1, 5, 10, 999], labels=['Low', 'Med', 'High']).astype(str)
    
    # Combine with Experience (Feature Cross)
    df['exp_skills_cross'] = df['experience_level'].astype(str) + "_" + df['skills_bin']
    
    return df

def get_feature_columns():
    """
    Returns the list of columns expected by the model pipeline.
    """
    return ['skills', 'qualification', 'experience_level', 'exp_skills_cross']
