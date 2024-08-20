# app.py
import os
import sys
import streamlit as st
import pandas as pd

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import train_model

# Train the model
model = train_model()

# Streamlit app
st.title("Iris Species Prediction")

st.write("""
## Input the features of the Iris flower:
""")

# Input fields for the features
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.2)

# Predict button
if st.button('Predict'):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)
    st.write(f"Predicted Iris Species: {prediction[0]}")
