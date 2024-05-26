import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def home():
    st.title("Home")
    st.write("Welcome to the home page!")

def data_exploration():
    st.title("Data Exploration")
    st.write("Explore your data here.")
    # Add your data exploration code here
    # Example: st.dataframe(df)

def model_training():
    st.title("Model Training")
    st.write("Train your model here.")
    # Add your model training code here
    # Example: st.button("Train Model")

def fraud_detection():
    st.title("Fraud Detection")
    st.write("Detect fraud here.")
    # Add your fraud detection code here
    # Example: st.button("Detect Fraud")

# Create a sidebar with navigation options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Training", "Fraud Detection"])

# Display the selected page
if page == "Home":
    home()
elif page == "Data Exploration":
    data_exploration()
elif page == "Model Training":
    model_training()
elif page == "Fraud Detection":
    fraud_detection()

