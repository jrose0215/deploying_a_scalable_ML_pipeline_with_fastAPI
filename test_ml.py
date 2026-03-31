import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference


def test_train_model_returns_classifier():
    """
    Test that train_model returns a RandomForestClassifier.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns correct precision, recall, and F1.
    """
    y = np.array([1, 0, 1, 1, 0])
    preds = np.array([1, 0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert 0 <= fbeta <= 1


def test_inference_output_shape():
    """
    Test that inference returns predictions with the same length as input.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    X_test = np.array([[1, 2], [3, 4]])
    preds = inference(model, X_test)
    assert len(preds) == len(X_test)