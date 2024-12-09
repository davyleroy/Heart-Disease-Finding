# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Convert categorical variables
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)