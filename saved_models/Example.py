# %%
import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns


# %% [markdown]
# ## Data Handling
# First thing is to read the data, analyse it, select features, and prepare it for training by scaling and splitting it.

# %%
df = pd.read_csv('Data/heart.csv')
print(df.head())

# %% [markdown]
# ## Summarise the data
# 
# Using the describe method to get a summary of the data to identify to perform Heart diagnosis, determine if there are missing values and get a sense of the scale of values in different columns.

# %%
df.describe()

# %%
df.head(5)

# %%
df.drop(columns=['id','dataset'], inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)


# %% [markdown]
# ## data encoding

# %%
print(df.head(2))

# %%
encoder = LabelEncoder()

df['sex'] = encoder.fit_transform(df['sex'])
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
df['cp'] = encoder.fit_transform(df['cp'])
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
df['fbs'] = encoder.fit_transform(df['fbs'])
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
df['restecg'] = encoder.fit_transform(df['restecg'])
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
df['exang'] = encoder.fit_transform(df['exang'])
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
df['slope'] = encoder.fit_transform(df['slope'])
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
df['thal'] = encoder.fit_transform(df['thal'])
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# %% [markdown]
# ### Deciding on important features
# Deciding on which features are most important to the target our main concern "Heart Disease" by plotting a correlation matrix. First we list all the columns in the dataset. Then we create a correlation matrix and lastly visualise the correlations using a heatmap.

# %%
columns = list(df.columns)
columns

# %% [markdown]
# ### Creating the correlation matrix
# 

# %%
corr = df.corr()
corr

# %% [markdown]
# ### Plotting a heatmap
# It seems there is no strong correlation between potability and individual features. There is need to explore data further to see if that is the case.

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, annot=True,)
plt.show()

# %% [markdown]
# ### Drawing scatter plots between different features
# Trying to determine if there are obvious trends in the way the data is spread in order to determine the Heart disease occurance.

# %%
# Create a list of columns to plot by copying the columns list
# And removing the Potability column
columns_to_plot = columns.copy()
columns_to_plot.remove('num')

for column in columns_to_plot:
  plt.figure(figsize=(13, 1))
  sns.scatterplot(x=df[column], y=df['num'])
  plt.title(f'Scatter plot of {column} vs num')
  plt.xlabel(column)
  plt.ylabel('Presence of heart disease')
  plt.show()

# %% [markdown]
# ### Conclusion
# 
# The implementation of AI models for heart disease diagnosis demonstrates significant potential for improving diagnostic accuracy and efficiency. As shown by the heatmap, a medium correlation was observed between various features and the target variable, highlighting the relationships that inform our model's predictions. Additionally, the scatterplots provided further insights into the distribution of data points, illustrating how different features interact with one another. By harnessing these visualizations alongside advanced machine learning techniques, we aim to revolutionize the diagnostic process, ultimately leading to better patient outcomes and more informed healthcare decisions.

# %% [markdown]
# ### Preparing the data for training
# To prepare the data for training, we are going to scale the data, split the data into features and target, and then split it further into training and testing data.

# %%
x = df.drop(columns=['num'])
y = df['num']

scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

print(x.head(5))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
print(x_train.shape)
print(y_train.shape)

# %% [markdown]
# # model architecture

# %% [markdown]
# ## Making the Vanilla Model
# 
# Training a model without regularisation or optimisation.

# %% [markdown]
# ## Defining the model
# 1. Create the model as a Sequential model.
# 
# 2.   Add layers to the model:
# 
# 
# *   Start by adding a dense layer with 16 units, ReLU activation function, and specify the input dimension to match the number of features in the training data.
# *   Add another dense layer with 32 units, ReLU activation function, and specify the input dimension to match the number of features in the training data.
# *   Add another dense layer with 1 unit and a sigmoid activation function. This layer will output the probability of water potability.
# 
# 

# %%
from keras.models import Sequential
from keras.layers import Dense


model1 = Sequential()
model1.add(Dense(128, activation="relu", input_shape=(x_train.shape[1],)))
model1.add(Dense(64, activation="relu"))
model1.add(Dense(64, activation="relu"))
model1.add(Dense(32, activation="relu"))
model1.add(Dense(1, activation="sigmoid"))
model1.summary() 


model1.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy','precision','recall'])


# %% [markdown]
# 3.   Compile the model:
# 
# *   Compile the model using the `adamax` optimizer.
# *   Specify the loss function as `binary_crossentropy` since this is a binary classification problem.
# *   Include `accuracy` as a metric to track during training.
# ### Icluding  : Training the model
# 1.   Train the model:
# 
# *   Use the `fit` method to train the model.
# *   Provide the training data (`X_train` and `Y_train`).
# *   Set the number of epochs to 100.
# *   Specify the batch size as 32.

# %%
model.compile(
    loss="binary_crossentropy", metrics=["accuracy", "Precision", "Recall"], optimizer="adamax"
)

model_train = model.fit(
    x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1
)

model1.save("saved_models/model1.keras")

# %% [markdown]
# ## optmisation

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1


model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(x_train.shape[1],), kernel_regularizer=l1(l1=0.001)))
model.add(Dropout(0.4))
model.add(Dense(64, activation="relu", kernel_regularizer=l1(l1=0.001)))
model.add(Dropout(0.4))
model.add(Dense(64, activation="relu",  kernel_regularizer=l1(l1=0.001)))
model.add(Dropout(0.4))
model.add(Dense(32, activation="tanh",  kernel_regularizer=l1(l1=0.001)))
model.add(Dropout(0.4))
model.add(Dense(1, activation="sigmoid"))
model.summary() 


model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy','precision','recall'])


# %%
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=15, mode='auto')


model.compile(
    loss="binary_crossentropy", metrics=["accuracy", "Precision", "Recall"], optimizer=Adam(learning_rate=0.005)
)

model_train = model.fit(
    x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stop]
)

# %% [markdown]
# ### Addition of an L1 Regularization to the Model
# 
# To prevent overfitting and improve generalization, we will implement the architecture used previously but with L1 regularization. We will also add early stopping to find the best weights and dropout to prevent overfitting.
# 
# #### steps taken
# 
# -   Defining the model with L1 regularization and dropout between layers.
# -   Training the model with early stopping

# %%
from keras.layers import BatchNormalization
model2 = Sequential()
model2.add(Dense(128, activation="relu", input_shape=(x_train.shape[1],), kernel_regularizer=l1(l1=0.001)))
model2.add(BatchNormalization())
model2.add(Dropout(0.4))
model2.add(Dense(32, activation="relu", kernel_regularizer=l1(l1=0.001)))
model2.add(Dense(64, activation="relu", kernel_regularizer=l1(l1=0.001)))
model2.add(Dense(1, activation="sigmoid", kernel_regularizer=l1(l1=0.001)))


model2.compile(
    loss="binary_crossentropy", metrics=["accuracy", "Precision", "Recall"], optimizer=Adam(learning_rate=0.001)
)

model_train = model2.fit(
    x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stop]
)

model2.save("saved_models/model2.keras")

# %% [markdown]
# ## Findings : Error Analysis
# 
# In this section we will compare the different models and how they perform so as to decide on the best model to use

# %% [markdown]
#    - **Performance Metrics Vanilla Model**:
#      - Loss: 0.0045
#      - Accuracy: 1.0000
#      - Precision: 1.0000
#      - Recall: 1.0000
#      - Validation Loss: 1.4946
#  - **Performance Metrics Optimized model**:
#      - Loss: 0.6184
#      - Accuracy: 0.9791
#      - Precision: 0.9796
#      - Recall: 0.9796
#      - Validation Loss: 1.0764

# %% [markdown]
# ### Error Analysis
# 
# In this section, we will analyze the errors made by our models to gain insights into their performance and identify areas for improvement.
# 
# 1. **Confusion Matrix**: 
#    We will visualize the confusion matrix for both models to understand the true positives, false positives, true negatives, and false negatives.

# %%
y_pred_simple = model1.predict(x_test)
y_pred_simple = (y_pred_simple > 0.5).astype(int)
cm_simple = confusion_matrix(y_test, y_pred_simple)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_simple, annot=True, fmt='d', cmap='Blues',
xticklabels=['No Heart Disease', 'Heart Disease'],
yticklabels=['No Heart Disease', 'Heart Disease'])
plt.title('Confusion Matrix - Simple Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %%
y_pred_optimized = model2.predict(x_test)
y_pred_optimized = (y_pred_optimized > 0.5).astype(int)
cm_optimized = confusion_matrix(y_test, y_pred_optimized)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Blues',
xticklabels=['No Heart Disease', 'Heart Disease'],
yticklabels=['No Heart Disease', 'Heart Disease'])
plt.title('Confusion Matrix - Optimized Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %% [markdown]
# 
# 2. **Error Analysis**: 
#    We will analyze the misclassified instances to understand the characteristics of the samples that were incorrectly predicted.
# 

# %% [markdown]
# ### Identify & display misclassified instances for the simple model  

# %%


# %%
# Ensure y_test is a NumPy array
y_test_values = y_test.values  # Convert to NumPy array if it's a DataFrame/Series

# Generate predictions using the trained model
y_pred_simple = model.predict(x_test)
y_pred_simple = (y_pred_simple > 0.5).astype(int).flatten()  # Flatten to ensure it's 1D

# Identify misclassified instances
misclassified_simple = x_test[y_test_values != y_pred_simple]  # Use boolean indexing
misclassified_simple = misclassified_simple.copy()  # Create a copy to avoid SettingWithCopyWarning
misclassified_simple['Actual'] = y_test_values[y_test_values != y_pred_simple]  # Add actual values
misclassified_simple['Predicted'] = y_pred_simple[y_test_values != y_pred_simple]  # Add predicted values

# Display misclassified instances
print("Misclassified Instances - Simple Model:")
print(misclassified_simple)


