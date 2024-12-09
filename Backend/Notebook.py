# %%
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# %%
df = pd.read_csv('heart.csv')

# %%
df.head()

# %%
#  drop id, dataset,
df = df.drop(['id', 'dataset'], axis=1)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# %%
df.head()

# %%
# encode, sex, cp, fbs, restecg, exang, slope, thal

labelencoder = LabelEncoder()

df["sex"] = labelencoder.fit_transform(df["sex"])
print(dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_))))
df["cp"] = labelencoder.fit_transform(df["cp"])
print(dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_))))
df["fbs"] = labelencoder.fit_transform(df["fbs"])
print(dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_))))
df["restecg"] = labelencoder.fit_transform(df["restecg"])
print(dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_))))
df["exang"] = labelencoder.fit_transform(df["exang"])
print(dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_))))
df["slope"] = labelencoder.fit_transform(df["slope"])
print(dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_))))
df["thal"] = labelencoder.fit_transform(df["thal"])
print(dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_))))

      


# %%
df.head()

# %%
scaler = StandardScaler()

# List of columns to scale
columns_to_scale = ["trestbps", "chol", "thalch", "age"]

# Apply the scaler to the selected columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

df.head()

# %%
import joblib

joblib.dump(scaler, 'scaler.pkl')

# %%
y = df["num"]
print(y_train.unique())  # Check if all labels are within range

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout

X = df.drop("num", axis=1)
y = df["num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()

model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# %%
model.fit(X_train, y_train, epochs=100, batch_size=10)

# %%
model.save('model.h5')
model.save('model.keras')

# %%
# 1.384143	1	0	1.596354	0.747722	0	0	-1.790447	1	1.5	1	3.0	1	2
test_data = np.array([[1.384143, 1, 0, 1.596354, 0.747722, 0, 0, -1.790447, 1, 1.5, 1, 3.0, 1]])
prediction = model.predict(test_data)

predicted_class = np.argmax(prediction)

print(predicted_class)

# %%



