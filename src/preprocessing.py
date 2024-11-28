import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

filepaths = {
    "train": "data/train.csv",
    "test": "data/test.csv"
}

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def encode_features(df, categorical_columns):
    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])
    return df

def scale_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)
