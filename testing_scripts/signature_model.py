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

    try:

        # Create a dummy input for the model
        input_text = "hi how are you"
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=vectorizer.get_feature_names_out())  # <-- Use correct feature names

        # Predict using the model
        prediction = model.predict(input_df)

        # Verify the input shape matches the vectorizer's feature output
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"

        # Verify the output shape (assuming binary classification with a single output)
        assert len(prediction) == input_df.shape[0], "Output rows count mismatch"

    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")

    print("Test passed: Input shape matches the model's expected number of columns.")


