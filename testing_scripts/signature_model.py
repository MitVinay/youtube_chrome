import pytest
import mlflow.pyfunc
import joblib
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_model_and_input_shape():
    """
    Combined test to load the model and vectorizer,
    ensure the dummy input matches the model's expected input shape,
    and validate predictions.
    """
    # Model and vectorizer details
    model_name = "Final_model"
    model_version = "1"
    vectorizer_path = "./tfidf_vectorizer.pkl"

    # Set tracking URI for MLflow
    tracking_uri = "https://dagshub.com/MitVinay/youtube_chrome.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"Set MLflow tracking URI to: {tracking_uri}")

    model_uri = f"models:/{model_name}/{model_version}"
    logger.info(f"Model URI: {model_uri}")

    # Load the model
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully.")
        logger.debug(f"Model type: {type(model)}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        pytest.fail(f"Model loading failed: {e}")

    # Load the vectorizer
    try:
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Vectorizer loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Vectorizer file not found at: {vectorizer_path}")
        pytest.fail("Vectorizer file not found.")
    except Exception as e:
        logger.error(f"Error loading vectorizer: {e}")
        pytest.fail(f"Vectorizer loading failed: {e}")

    # Create dummy input and test
    try:
        input_text = "hi how are you"
        logger.info(f"Input text: {input_text}")

        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=vectorizer.get_feature_names_out())
        logger.info(f"Transformed input shape: {input_df.shape}")

        # Predict using the model
        prediction = model.predict(input_df)
        logger.info(f"Prediction: {prediction}")

        # Verify input shape matches vectorizer output
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"
        logger.info("Input shape matches vectorizer feature count.")

        # Verify output shape
        assert len(prediction) == input_df.shape[0], "Output rows count mismatch"
        logger.info("Output shape matches input rows count.")

    except Exception as e:
        logger.error(f"Error during prediction or validation: {e}")
        pytest.fail(f"Model test failed during prediction or validation: {e}")

    logger.info("Test passed: Model and input shape are correct.")
