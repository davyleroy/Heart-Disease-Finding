from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from preprocessing import preprocess_data, split_data
from model import train_model, save_model, load_model
from evaluation import evaluate_model
from keras.models import load_model
import joblib
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = load_model('../saved_models/heart_precision.keras')
scaler = None

def initialize_model_and_scaler():
    global model, scaler
    model_path = '../saved_models/heart_precision.h5'
    scaler_path = '../saved_models/scaler.pkl'
    
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Please train the model first.")
    
    try:
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
    except FileNotFoundError:
        print(f"Scaler file not found at {scaler_path}. It will be created during the first training or prediction.")

def train_and_save_model():
    global model, scaler
    # Load your dataset here
    df = pd.read_csv('path_to_your_dataset.csv')
    
    # Preprocess data
    X_scaled, y, scaler = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # Train model
    model, history = train_model(X_train, y_train, X_test, y_test)
    
    # Save model and scaler
    model_path = '../saved_models/heart_precision.h5'
    scaler_path = '../saved_models/scaler.pkl'
    save_model(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Model and scaler saved successfully.")

initialize_model_and_scaler()

@app.post('/train')
async def train():
    train_and_save_model()
    return {'message': 'Model trained and saved successfully'}

@app.post('/predict')
async def predict(data: dict):
    global model, scaler
    if model is None:
        raise HTTPException(status_code=400, detail='Model not trained')
    
    df = pd.DataFrame([data])
    
    if scaler is None:
        # If scaler is not loaded, use a default StandardScaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(df)
        joblib.dump(scaler, '../saved_models/scaler.pkl')
    
    X_scaled = scaler.transform(df)
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)[0][1]
    
    return {
        'prediction': int(prediction[0]),
        'probability': float(probability)
    }

@app.post('/upload')
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Invalid file format')
    
    df = pd.read_csv(file.file)
    return {'message': 'File uploaded successfully', 'rows': len(df)}

@app.post('/retrain')
async def retrain(file: UploadFile = File(...)):
    global model, scaler
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Invalid file format')
    
    df = pd.read_csv(file.file)
    
    # Preprocess data
    X_scaled, y, new_scaler = preprocess_data(df)
    
    # Update the global scaler
    scaler = new_scaler
    joblib.dump(scaler, '../saved_models/scaler.pkl')
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model)
    
    return {
        'message': 'Model retrained successfully',
        'metrics': metrics
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
