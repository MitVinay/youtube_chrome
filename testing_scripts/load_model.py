import mlflow
import joblib
import pytest

def load_model_and_vectorizer():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("https://dagshub.com/MitVinay/youtube_chrome.mlflow")  # Replace with your MLflow URI

    # Model details
    model_name = "Final_model"  # Replace with the actual model name in MLflow
    model_version = "1"  # Replace with the desired model version

    # Construct the model URI
    model_uri = f"models:/{model_name}/{model_version}"
    print(f"Model URI: {model_uri}")

    # Load the model using MLflow
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

    # Load the vectorizer from local storage
    vectorizer_path = "./tfidf_vectorizer.pkl"  # Path to the vectorizer
    try:
        vectorizer = joblib.load(vectorizer_path)
        print("Vectorizer loaded successfully.")
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        vectorizer = None

    return model, vectorizer

@pytest.fixture
def model_and_vectorizer():
    # Load the model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    return model, vectorizer

def test_model_loading(model_and_vectorizer):
    model, vectorizer = model_and_vectorizer
    
    # Test if the model is loaded properly
    assert model is not None, "Model failed to load"
    
    # Test if the vectorizer is loaded properly
    assert vectorizer is not None, "Vectorizer failed to load"
    
    # You can also check specific types if needed, e.g.:
    assert hasattr(model, "predict"), "Model does not have the 'predict' method"
    assert hasattr(vectorizer, "transform"), "Vectorizer does not have the 'transform' method"

    print("Model and vectorizer loaded successfully!")

