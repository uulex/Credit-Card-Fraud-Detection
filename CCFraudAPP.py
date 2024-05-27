import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


data = pd.read_csv("creditcard_2023.csv")

def home():
    st.title("Credit Card Fraud Detection")
    st.subheader("Motivation")
    st.write("Our team took on the task to develop a ML-modell which given certain parameters of a credit card transaction could determine wether a transaction was fraudulent or not. ")

def data_exploration():
    st.title("Data Exploration")
    st.subheader("Raw Data")
    st.markdown("""
    The columns of our dataset consist of <span style="color: light blue;">28 anonymized Parameters</span>, 
    the <span style="color: light blue;">transaction-amount in $</span>, a  
    <span style="color: light blue;">unique Id</span>, and the <span style="color: light blue;">Class</span> which determines 
    if a transaction was <span style="color: light blue;">fraudulent or not</span> (1 = fraudulent; 0 = not fraudulent). 
    The <span style="color: light blue;">total number of rows</span> in our dataset is <span style="color: light blue;">568630</span>.
    """, unsafe_allow_html=True)
    with st.expander("Show raw data"):
        st.write(data.shape)
        st.write(data.head())

    st.subheader("Correlation Matrix")
    st.markdown("""The highest correlation between parameters are between <span style="color: light blue;">V16, V17</span> and <span style="color: light blue;">V18</span>.""", unsafe_allow_html=True)
    heatmap = plt.figure(figsize=[20,10])
    sns.heatmap(data.corr(),cmap="crest", annot=True)
    with st.expander("Show correlation heatmap"):
        st.pyplot(heatmap)

    st.subheader("Distribution of amount-parameter")
    st.markdown("""The <span style="color: light blue;">amounts</span> of money transferred in the transactions contained in or data set are <span style="color: light blue;">evenly distributed</span>.""", unsafe_allow_html=True)





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

#Text at bottom of sidebar
st.sidebar.markdown(
    """
    <hr style="margin-top: 20px; margin-bottom: 10px;">
    <p style="font-size: 12px;">An App Created by Jeremi Degenhardt, Frederic von Gahlen, Leo Gfeller and Alexander Nigg</p>
    """,
    unsafe_allow_html=True
)
