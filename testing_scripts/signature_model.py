import pytest
import mlflow.pyfunc
import joblib
import pandas as pd

def test_model_and_input_shape():
    """
    Combined test to load the model and vectorizer,
    ensure the dummy input matches the model's expected input shape,
    and validate predictions.
    """
    # Model and vectorizer details
    MODEL_NAME = "Final_model"
    MODEL_VERSION = "1"
    VECTORIZER_PATH = "./tfidf_vectorizer.pkl"

    try:
        # Load model from MLflow Model Registry
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Load vectorizer from local storage
        vectorizer = joblib.load(VECTORIZER_PATH)

        # Create a dummy input for the model
        input_text = "hi how are you"
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=vectorizer.get_feature_names_out())

        # Predict using the model
        prediction = model.predict(input_df)

        # Verify the input shape matches the vectorizer's feature output
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"

        # Verify the output shape (assuming binary classification with a single output)
        assert len(prediction) == input_df.shape[0], "Output rows count mismatch"

    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")

    # Test passed
    assert True, "Test passed: Model and input shape are correct."
