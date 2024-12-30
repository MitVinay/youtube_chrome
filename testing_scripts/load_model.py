# load_model.py
import joblib

def load_model(model_path='lgbm_model.pkl'):
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

if __name__ == "__main__":
    model = load_model()
