from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import joblib

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Loading model and scaler...")
        yield
    finally:
        await app.state.shutdown()
        await app.state.cleanup()

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained model and scaler
model = load_model("model.keras")  # Replace with your model file path
scaler = joblib.load("scaler.pkl")

# Define a simple route
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Heart Disease Prediction API"}

# Label encoders
label_encoders = {
    "sex": {"Female": 0, "Male": 1},
    "cp": {
        "asymptomatic": 0,
        "atypical angina": 1,
        "non-anginal": 2,
        "typical angina": 3,
    },
    "fbs": {False: 0, True: 1},
    "restecg": {"lv hypertrophy": 0, "normal": 1, "st-t abnormality": 2},
    "exang": {False: 0, True: 1},
    "slope": {"downsloping": 0, "flat": 1, "upsloping": 2},
    "thal": {"fixed defect": 0, "normal": 1, "reversable defect": 2},
}

# Columns to scale
columns_to_scale = ["trestbps", "chol", "thalch", "age"]


# Input data model
class HeartDiseaseData(BaseModel):
    age: float
    sex: str
    cp: str
    trestbps: float
    chol: float
    fbs: bool
    restecg: str
    thalch: float
    exang: bool
    oldpeak: float
    slope: str
    ca: float
    thal: str


@app.post("/predict")
def predict(data: HeartDiseaseData):
    # Convert input to a DataFrame
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # Apply label encoding
    for column, mapping in label_encoders.items():
        if column in df:
            df[column] = df[column].map(mapping)

    # Scale the numeric columns
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    # Convert to numpy array for prediction
    input_array = df.values

    # Make prediction
    prediction = model.predict(input_array)
    predicted_class = np.argmax(prediction)

    return {"predicted_class": int(predicted_class)}
