from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np
from .utils import preprocess_input, load_scaler

# Initialize blueprint
api_routes = Blueprint("api_routes", _name_)

# Load models
life_event_model = tf.keras.models.load_model("models/life_event_model.h5")
loan_model = tf.keras.models.load_model("models/loan_recommendation_model.h5")

# Load scaler
scaler = load_scaler("models/scaler.pkl")

@api_routes.route("/predict_event", methods=["POST"])
def predict_event():
    """
    Predict life events based on user financial behavior.
    """
    data = request.json
    sequence = preprocess_input(data["sequence"], scaler, seq_length=12)  # Assume 12 months
    prediction = life_event_model.predict(sequence)
    return jsonify({"probability": prediction[0][0]})

@api_routes.route("/recommend_loan", methods=["POST"])
def recommend_loan():
    """
    Recommend loans based on predicted life events and user demographics.
    """
    user_id = request.json["user_id"]
    loan_id = request.json["loan_id"]
    suitability = loan_model.predict([[user_id], [loan_id]])
    return jsonify({"suitability_score": suitability[0][0]})