import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

pickle_in = open("classifier.pkl", "rb")
model = pickle.load(pickle_in)

st.title("Wine Quality Prediction")

st.sidebar.header("Input Features")
def user_input_features():
    fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 8.0)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.08, 1.58, 0.5)
    citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.3)
    residual_sugar = st.sidebar.slider("Residual Sugar", 0.6, 15.5, 5.0)
    chlorides = st.sidebar.slider("Chlorides", 0.009, 0.611, 0.08)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 1, 72, 30)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 6, 289, 120)
    density = st.sidebar.slider("Density", 0.990, 1.004, 0.996)
    pH = st.sidebar.slider("pH", 2.74, 4.01, 3.3)
    sulphates = st.sidebar.slider("Sulphates", 0.33, 2.0, 0.75)
    alcohol = st.sidebar.slider("Alcohol", 8.4, 14.9, 10.5)
    
    data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

st.subheader("User Input Features")
st.write(input_df)


prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
wine_quality = "Good Quality" if prediction[0] == 1 else "Low Quality"
st.write(f"The predicted wine quality is: **{wine_quality}**")

st.subheader("Prediction Probability")
st.write(f"Probability of being Good Quality: {prediction_proba[0][1]:.2f}")
st.write(f"Probability of being Low Quality: {prediction_proba[0][0]:.2f}")
