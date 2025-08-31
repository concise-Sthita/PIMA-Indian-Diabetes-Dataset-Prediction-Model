# PIMA-Indian-Diabetes-Dataset-Prediction-Model
# Diabetes Risk Prediction Web App

This project develops a machine learning-powered web application that predicts a patient's risk of diabetes. The application, built with Python and Streamlit, uses an ensemble of both deep learning and traditional machine learning models for robust and comparative risk assessment.

## Objective

The primary goal was to create a user-friendly tool for clinicians to input patient health metrics and receive a diabetes risk prediction. The project also served as a deep dive into data preprocessing challenges and the comparative performance of different model architectures.

## Project Structure

* `diabetes.csv`: The raw PIMA Indians Diabetes Dataset.
* `diabetes_imputed.csv`: The cleaned dataset after handling missing values.
* `train_models.py`: A Python script containing all the data preprocessing, model training, and evaluation code.
* `streamlit_app.py`: The main script for the Streamlit web application.
* `model.keras`: The trained Artificial Neural Network (ANN) model.
* `scaler.pkl`: The trained `StandardScaler` object used for data normalization.
* `rf_model.pkl`: The trained Random Forest Classifier model.
* `lr_model.pkl`: The trained Logistic Regression model.
* `svm_model.pkl`: The trained Support Vector Machine (SVM) model.
* `xgb_model.json`: The trained XGBoost model.

## Methodology and Key Decisions

### 1. Data Preprocessing and Imputation Challenges

The PIMA Indians Diabetes Dataset presented a significant challenge: a large number of zero values in several key columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`). For these features, a value of zero is biologically impossible, indicating missing data.

-   **`Insulin` and `SkinThickness`**: These columns had the highest number of zeros (374 and 227, respectively), making a simple mean imputation highly problematic. A naive approach would have introduced significant bias, artificially compressing the data distribution and misleading the models.
-   **`Pregnancies` and `Outcome`**: We made a key distinction here. A value of zero in these columns is a valid data point, representing a person with no pregnancies or a non-diabetic outcome. Therefore, these columns were not imputed.

#### Our Solution: KNN Imputation

To overcome this, we used **K-Nearest Neighbors (KNN) Imputation**. This approach replaces missing values with the average of the k-nearest data points in the feature space. This is a superior method to mean imputation as it leverages correlations between features, leading to a more plausible and statistically sound distribution of imputed values.

### 2. Model Development and Comparison

The project's goal was to compare a deep learning model with several traditional machine learning models. We trained and evaluated five different classifiers:

-   **Artificial Neural Network (ANN)**: Our deep learning model, featuring two hidden layers and Dropout regularization to prevent overfitting.
-   **Random Forest Classifier**: A powerful ensemble model used as a baseline.
-   **Logistic Regression**: A simple but effective linear model for binary classification.
-   **Support Vector Machine (SVM)**: A robust classifier known for finding optimal decision boundaries, especially with scaled data.
-   **XGBoost**: A highly performant Gradient Boosting algorithm that consistently excels in tabular data competitions.

Each model was evaluated using a comprehensive set of metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### 3. Streamlit Web Application

The final deliverable is a functional web application for real-time predictions.

-   **User-Friendly Interface**: The app features a clean layout with interactive sliders for patient inputs. A key challenge was linking the sliders with text input fields to allow for both graphical and precise numerical input. This was solved using Streamlit's `session_state` to ensure two-way synchronization.
-   **Comprehensive Predictions**: The app loads all five trained models and provides a prediction and a confidence score for each, allowing users to see the results from different modeling approaches.
-   **Contextual Information**: An "About" section was added to provide transparency and context. It explains the project's methodology, data challenges, and the solutions implemented, including visual confirmation of our data imputation process.

## Setup and Installation

1.  **Clone the Repository**

    ```
    git clone [your_repository_link]
    cd [your_repository_name]
    
    ```

2.  **Install Dependencies**

    ```
    pip install pandas numpy scikit-learn tensorflow keras streamlit joblib xgboost
    
    ```

3.  **Run the Training Script**

    Execute the `train_models.py` script. This will perform all data preprocessing, model training, and save the necessary files (`.keras`, `.pkl`, `.json`) that the web app needs.

    ```
    python train_models.py
    
    ```

4.  **Run the Streamlit App**

    Once the training is complete, start the web application.

    ```
    streamlit run streamlit_app.py
    
    ```

    If the `streamlit` command is not recognized, use `python -m streamlit run streamlit_app.py`. The app will open in your default web browser.
