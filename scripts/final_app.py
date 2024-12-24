from flask import Flask, request, jsonify
import mlflow.pyfunc
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

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

# Define a route for the API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the raw input text from the POST request
    data = request.get_json()
    raw_text = data.get('comment')

    if not raw_text:
        return jsonify({'error': 'No comment provided'}), 400

    # Convert the raw text into a sparse matrix using the vectorizer
    vectorized_data = vectorizer.transform([raw_text])

    # Convert sparse matrix to a dense DataFrame (same structure as used during training)
    vectorized_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names_out())

    # Predict using the loaded model
    predictions = model.predict(vectorized_df)

    # Return predictions as a JSON response
    return jsonify({'prediction': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=8080)