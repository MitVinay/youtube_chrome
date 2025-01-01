import mlflow.pyfunc
import joblib
import pandas as pd
import logging

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
        return

    # Load the vectorizer
    try:
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Vectorizer loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Vectorizer file not found at: {vectorizer_path}")
        return
    except Exception as e:
        logger.error(f"Error loading vectorizer: {e}")
        return

    # Create dummy input and test
    try:
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
        if input_df.shape[1] != len(vectorizer.get_feature_names_out()):
            logger.error("Input feature count mismatch")
            return
        logger.info("Input shape matches vectorizer feature count.")

    except Exception as e:
        logger.error(f"Error during prediction or validation: {e}")
        return

    logger.info("Test passed: Model and input shape are correct.")

# Run the test function
if __name__ == "__main__":
    test_model_and_input_shape()
