import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

# --- Loading the trained models and scaler from saved files ---
try:
    # Load all models and the scaler
    ann_model = load_model('model.keras')
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    rf_model = joblib.load('rf_model.pkl')
    lr_model = joblib.load('lr_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    xgb_model = XGBClassifier()
    xgb_model.load_model('xgb_model.json')
except FileNotFoundError:
    st.error("Model files not found. Please ensure all model files ('model.keras', 'scaler.pkl', 'rf_model.pkl', 'lr_model.pkl', 'svm_model.pkl', 'xgb_model.json') are in the same directory.")
    st.stop()

# --- Title and Tabs ---
st.title("Diabetes Risk Prediction Web App")
st.markdown("Enter patient health details below to get a diabetes risk prediction.")
tab1, tab2 = st.tabs(["Prediction", "About"])

# --- Prediction Tab ---
with tab1:
    st.header("Predict Diabetes Risk")

    # --- Sidebar for Input Fields ---
    with st.sidebar:
        st.header("Patient Health Details")

        # Function to create a slider with a linked number input using session state
        def slider_with_input(label, min_val, max_val, default_val, step=1):
            slider_key = f"{label}_slider"
            input_key = f"{label}_input"

            if slider_key not in st.session_state:
                st.session_state[slider_key] = default_val
            if input_key not in st.session_state:
                st.session_state[input_key] = default_val

            def on_slider_change():
                st.session_state[input_key] = st.session_state[slider_key]

            def on_input_change():
                st.session_state[slider_key] = st.session_state[input_key]
            
            slider_val = st.slider(label, min_val, max_val, st.session_state[slider_key], step, key=slider_key, on_change=on_slider_change)
            st.number_input(f"Enter {label} (type to adjust)", min_val, max_val, st.session_state[input_key], step, key=input_key, on_change=on_input_change)
            
            # The function now correctly returns the value from the number input
            return st.session_state[input_key] 

        pregnancies = slider_with_input("Pregnancies", 0, 17, 1)
        glucose = slider_with_input("Glucose (mg/dL)", 0.0, 200.0, 120.0, 0.1)
        blood_pressure = slider_with_input("Blood Pressure (mm Hg)", 0.0, 122.0, 70.0, 0.1)
        skin_thickness = slider_with_input("Skin Thickness (mm)", 0.0, 100.0, 20.0, 0.1)
        insulin = slider_with_input("Insulin (mu U/ml)", 0.0, 846.0, 79.0, 0.1)
        bmi = slider_with_input("BMI", 0.0, 70.0, 30.0, 0.1)
        pedigree = slider_with_input("Diabetes Pedigree Function", 0.0, 3.0, 0.471, 0.001)
        age = slider_with_input("Age", 0, 120, 30)

    # --- Prediction Button and Results ---
    if st.button("Predict Diabetes Risk"):
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]],
                                  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        scaled_input = scaler.transform(input_data)
        
        # Make predictions with all models
        ann_proba = ann_model.predict(scaled_input)[0][0]
        rf_proba = rf_model.predict_proba(scaled_input)[:, 1][0]
        lr_proba = lr_model.predict_proba(scaled_input)[:, 1][0]
        svm_proba = svm_model.predict_proba(scaled_input)[:, 1][0]
        xgb_proba = xgb_model.predict_proba(scaled_input)[:, 1][0]
        
        st.subheader("Prediction Results")

        def display_prediction(model_name, proba):
            risk = "High Risk" if proba >= 0.5 else "Low Risk"
            color = "red" if risk == "High Risk" else "green"
            st.markdown(f"**{model_name} Prediction:** <span style='color:{color}'>**{risk}**</span>", unsafe_allow_html=True)
            st.write(f"Confidence Score: {proba:.2f}")

        display_prediction("ANN", ann_proba)
        display_prediction("Random Forest", rf_proba)
        display_prediction("Logistic Regression", lr_proba)
        display_prediction("SVM", svm_proba)
        display_prediction("XGBoost", xgb_proba)
        
        st.success("Prediction complete!")

# --- About Tab ---
with tab2:
    st.link_button("View Dataset on Kaggle", "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
    st.link_button("View My Resume", "https://drive.google.com/drive/folders/1IjjM7QQbBA47cJjmwVKybxutQEGCTDl2?usp=sharing")
    st.link_button("View Project Code on GitHub", "https://github.com/concise-Sthita/PIMA-Indian-Diabetes-Dataset-Prediction-Model")
    st.link_button("View My LinkedIn", "https://www.linkedin.com/in/sthitapragyan-mahapatra/")
    st.header("About the Model")
    st.write("This web application provides a diabetes risk prediction based on a patient's health metrics. The prediction engine is powered by a deep learning model (ANN) and several traditional machine learning models for comparison.")
    st.write("---")  
    st.subheader("Our Approach: From Raw Data to Predictions")
    st.write("The core of this project was addressing the challenges within the dataset to build a robust and accurate model.")
    st.write("1. **Data Challenges**: The PIMA Indians Diabetes Dataset contained a significant number of zero values for key health metrics like Insulin and Glucose. A value of 0 for these features is biologically impossible, indicating missing data.")
    st.image("insulin_imputation_comparison.png", caption="Distribution of Insulin before Imputation (Note the large spike at 0 in the before graph) and after KNN Imputation")
    st.write("2. **Overcoming Challenges**: A simple fix like mean imputation would have severely skewed the model, especially for the Insulin column where nearly half the values were missing. To handle the missing data more accurately, we used a **K-Nearest Neighbors (KNN) imputation** technique. This approach replaced missing values with a statistically plausible estimate based on the values of the most similar patients in the dataset. This allowed us to preserve the underlying relationships between features and avoid biasing the model.")
    st.image("Pregnancies_imputation_comparison.png","0 for Pregnancies is a valid data point,\nrepresenting a person who has had no pregnancies,\nIn contrast, a value of 0 for features like Glucose or BMI\nis biologically impossible and must be treated as a missing value")
    st.write("3. **Model Architecture**: We explored multiple algorithms for this task. Our deep learning model is an **Artificial Neural Network (ANN)** with two hidden layers and a **Dropout regularization** to prevent the model from overfitting. We also built and compared a **Random Forest Classifier** and a **Logistic Regression** model as a baseline. For a more robust comparison, we also implemented powerful gradient boosting algorithms like **XGBoost** and a **Support Vector Machine (SVM)** model.")

    st.write("4. **Results**: We chose the final models based on a comprehensive evaluation of metrics including Accuracy, Precision, Recall, F1-Score, and ROC-AUC. This confirmed that our imputation and model design choices led to a robust and reliable predictor.")
