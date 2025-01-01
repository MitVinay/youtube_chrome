import pytest
import mlflow
import numpy as np
import pandas as pd
import joblib

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = "https://dagshub.com/MitVinay/youtube_chrome.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Model and vectorizer details
MODEL_NAME = "Final_model"
MODEL_VERSION = "1"
VECTORIZER_PATH = "./tfidf_vectorizer.pkl"

@pytest.fixture
def load_model_and_vectorizer():
    """
    Fixture to load the model and vectorizer.
    """
    # Load model from MLflow Model Registry
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Load vectorizer from local storage
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer

def test_input_shape(load_model_and_vectorizer):
    """
    Test to ensure the dummy input matches the model's expected input shape.
    """
    model, vectorizer = load_model_and_vectorizer

    # Create a dummy input
    dummy_input = ["This is a test comment for validation."]

    # Preprocess the input using the vectorizer
    transformed_input = vectorizer.transform(dummy_input)

    # Convert sparse matrix to dense DataFrame (as expected by the model)
    input_df = pd.DataFrame(transformed_input.toarray(), columns=vectorizer.get_feature_names_out())

    # Fetch the number of columns expected by the model
    expected_num_columns = model.metadata.get_input_schema().num_columns

    # Assert that the input DataFrame has the same number of columns as the model expects
    assert input_df.shape[1] == expected_num_columns, (
        f"Input shape mismatch: Expected {expected_num_columns} columns, but got {input_df.shape[1]}."
    )

    print("Test passed: Input shape matches the model's expected number of columns.")
