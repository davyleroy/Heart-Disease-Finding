import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.regularizers import l1
import joblib
import os
from sklearn.model_selection import train_test_split

def save_model(model, model_path):
    model.save(model_path, save_format='h5')

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def build_model(input_shape):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_shape,), kernel_regularizer=l1(0.001)),
        Dropout(0.5),
        Dense(64, activation="relu", kernel_regularizer=l1(0.001)),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    return model

def train_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],), kernel_regularizer=l1(0.001)),
        Dropout(0.5),
        Dense(64, activation="relu", kernel_regularizer=l1(0.001)),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

    return model, history
