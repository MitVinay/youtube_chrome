import mlflow.pyfunc
import pickle
import pandas as pd

# Set the MLflow tracking URI (Ensure that the experiment is properly set up)
mlflow.set_tracking_uri("https://dagshub.com/MitVinay/youtube_chrome.mlflow")

# Specify the model name registered in MLflow
model_name = "yt_chrome_plugin_model"

# Specify the model version (ensure the version exists in the MLflow registry)
model_version = 1

# Load the model from MLflow
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

# Load the TF-IDF vectorizer using pickle (for consistency)
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Raw input text
raw_text = ["You are good"]

# Convert the raw text into a sparse matrix using the vectorizer
vectorized_data = vectorizer.transform(raw_text)

# Convert sparse matrix to a dense DataFrame (same structure as used during training)
vectorized_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names_out())

# Predict using the loaded model
predictions = model.predict(vectorized_df)

# Print the predictions
print(predictions)
