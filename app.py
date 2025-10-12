import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import os

model_path = os.path.join(os.path.dirname(__file__), 'hero_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

st.title("Superhero Recruit Classifier")
st.write("Enter the recruit's attributes to classify them as a hero or not.")

# User inputs
strength = st.slider("Strength (0-10)", min_value=0, max_value=10, value=5)
speed = st.slider("Speed (0-10)", min_value=0, max_value=10, value=5)
intelligence = st.slider("Intelligence (0-10)", min_value=0, max_value=10, value=5)

# Predict Button
if st.button("Classify Recruit"):
    new_recruit  = np.array([[strength, speed, intelligence]])
    prediction = model.predict(new_recruit)[0]

    if prediction == 1:
        st.success("Hero! Welcome to the team!")
    else:
        st.error("Villain! You cannot join the team.")