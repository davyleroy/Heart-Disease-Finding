# Heart-Disease-Finding

## Project Overview

The primary goal of this project is to enhance the diagnosis of heart diseases through the application of advanced AI models. By leveraging machine learning techniques, we aim to provide faster and more accurate assessments of heart health, ultimately improving patient outcomes.

## Objectives

- **Improve Diagnosis Accuracy**: Utilize AI algorithms to analyze patient data and identify potential heart disease indicators with higher precision than traditional methods.
- **Speed Up Diagnosis Process**: Reduce the time required for diagnosis by automating data analysis, allowing healthcare professionals to focus on patient care.

- **Data-Driven Insights**: Generate actionable insights from large datasets, helping to inform treatment decisions and preventive measures.

## AI Model

This notebook contains the implementation of the AI model designed for heart disease diagnosis. It includes data preprocessing, model training, and evaluation steps, showcasing how AI can be integrated into healthcare practices.

## Model Implementation

We implemented two models to evaluate their performance:

1. **Simple Machine Learning Model**:

   - A basic neural network model was applied to the chosen dataset without any optimization techniques.
   - **Performance Metrics**:
     - Loss: 0.0045
     - Accuracy: 1.0000
     - Precision: 1.0000
     - Recall: 1.0000
     - Validation Loss: 1.4946

2. **Optimized Model**:
   - This model applied at least three optimization techniques: normalization, L1 regularization, and early stopping.
   - **Performance Metrics**:
     - Loss: 0.6184
     - Accuracy: 0.9791
     - Precision: 0.9796
     - Recall: 0.9796
     - Validation Loss: 1.0764

### Libraries Used

The following libraries were utilized in the implementation:

### Parameter Settings

- **Data Preprocessing**:

  - The dataset was split into training and testing sets using `train_test_split`.
  - Features were standardized using `StandardScaler` to improve model performance.
  - Categorical variables were encoded using `LabelEncoder`.

- **Model Architecture**:

  - The simple model consisted of a basic feedforward neural network.
  - The optimized model included additional layers and applied L1 regularization to prevent overfitting.

- **Training Parameters**:
  - Batch size, learning rate, and number of epochs were tuned based on validation performance.

### Observed Results

- A medium correlation was observed in the heatmap generated from the dataset, indicating relationships between various features and the target variable. This correlation analysis helped in understanding the importance of different features in predicting heart disease.

- The confusion matrix was plotted to visualize the performance of the models, providing insights into true positives, false positives, true negatives, and false negatives.

## Instructions for Running the Notebook

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/davyleroy/Heart-Disease-Finding.git
   cd Heart-Disease-Finding
   ```

2. **Install Required Libraries**:
   Ensure you have the necessary libraries installed. You can use pip to install them:

   ```bash
   pip install tensorflow pandas numpy matplotlib scikit-learn seaborn
   ```

3. **Run the Notebook**:
   Open the `notebook.ipynb` file in Jupyter Notebook or any compatible environment and execute the cells to run the analysis.

## Loading Saved Models

To load the saved models, you can use the following code snippet in your notebook:

```python
python
import joblib
Load the models
model1 = joblib.load('saved_models/model1.pkl')
model2 = joblib.load('saved_models/model2.pkl')
```

### Conclusion

By harnessing the power of AI, we aim to revolutionize the way heart diseases are diagnosed, making the process more efficient and reliable for healthcare providers and patients alike.
