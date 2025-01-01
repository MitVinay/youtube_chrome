import pytest
import mlflow.pyfunc
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def model_and_vectorizer():
    """
    Fixture to load the model and vdectorizer once for the entire test module.
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

    return model, vectorizer

def test_model_and_input_shape(model_and_vectorizer):
    """
    Test to ensure the model and input shape are correct.
    """
    model, vectorizer = model_and_vectorizer

    # Create dummy input and test
    input_text = ["hi how are you"]
    logger.info(f"Input text: {input_text}")
    
    transformed_comments = vectorizer.transform(input_text)
    input_df = pd.DataFrame(transformed_comments.toarray(), columns=vectorizer.get_feature_names_out())
    logger.info(f"Transformed input shape: {input_df.shape}")
    logger.info(f"Vectorizer input shape: {len(vectorizer.get_feature_names_out())}")

    # Predict using the model
    predictions = model.predict(input_df).tolist()
    logger.info(f"Prediction: {predictions}")

    # Verify input shape matches vectorizer output
    assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"
    logger.info("Input shape matches vectorizer feature count.")

    logger.info("Test passed: Model and input shape are correct.")

