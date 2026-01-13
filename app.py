from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Enable CORS (allow all origins for now)
CORS(app)

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return "Autism Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = data["features"]

        # Convert to numpy array
        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Render requires this
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
