import tensorflow as tf
import joblib
import pandas as pd

def predict(features):
    model = tf.keras.models.load_model("../saved_models/heart_disease_model.tf")
    scaler = joblib.load("../saved_models/scaler.pkl")
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return {"Prediction": "Heart Disease" if prediction[0][0] > 0.5 else "No Heart Disease"}

def retrain(new_data_path):
    from preprocessing import preprocess_data, split_data
    from model import train_model

    X, y = preprocess_data(new_data_path)
    model = train_model(X, y)
    return model
