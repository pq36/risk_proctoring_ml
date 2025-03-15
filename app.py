import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# Load the trained neural network model
model = load_model("nn_model.h5")

# Load the scaler and feature names from the pickle file
scaler_data = pickle.load(open("nn_scaler.pkl", "rb"))
scaler = scaler_data["scaler"]
FEATURE_NAMES = scaler_data["feature_names"]

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        # Replace missing values (None) with 0 (or any default value)
        for key in data:
            if data[key] is None:
                data[key] = 0

        # Convert input data to a DataFrame with the expected column order
        df_input = pd.DataFrame([data], columns=FEATURE_NAMES)
        
        # Scale the input features using the loaded scaler
        X_scaled = scaler.transform(df_input)
        
        # Predict the risk score using the neural network model
        predicted_array = model.predict(X_scaled)
        predicted_value = predicted_array[0][0]  # Extract scalar value
        
        # Clamp the risk score between 0 and 100 and round it to 2 decimals
        risk_score = max(0, min(100, round(predicted_value, 2)))
        
        return jsonify({"risk_score": risk_score})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=7000)
