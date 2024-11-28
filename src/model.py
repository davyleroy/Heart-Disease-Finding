import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.regularizers import l1
import joblib

def build_model(input_shape):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_shape,), kernel_regularizer=l1(0.001)),
        Dropout(0.4),
        Dense(64, activation="relu", kernel_regularizer=l1(0.001)),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(X_train, y_train):
    model = build_model(X_train.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])
    model.save("models/heart_disease_model.tf")
    return model

def save_model(model, scaler):
    model.save("models/heart_disease_model.tf")
    joblib.dump(scaler, "models/scaler.pkl")
