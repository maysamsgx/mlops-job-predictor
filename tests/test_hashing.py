"""
Unit tests for HashingVectorizer usage.
Validates that the hashing mechanism works for skills encoding.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

def test_hashing_vectorizer_output_shape():
    """
    Test that HashingVectorizer produces the expected number of features.
    """
    corpus = [
        'Python, SQL, TensorFlow',
        'Java, Spring',
        'Python, Docker, Kubernetes'
    ]

    n_features = 10
    vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False, norm=None)
    X = vectorizer.transform(corpus)

    assert X.shape == (3, n_features)
    assert X.dtype == np.float64

def test_hashing_consistency():
    """
    Test that the same input produces the same hash vector.
    """
    skills = ['Python, SQL']
    vectorizer = HashingVectorizer(n_features=20, alternate_sign=False, norm=None)

    hash1 = vectorizer.transform(skills).toarray()
    hash2 = vectorizer.transform(skills).toarray()

    np.testing.assert_array_equal(hash1, hash2)
