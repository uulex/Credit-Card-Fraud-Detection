import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

st.title('Credit Card Fraud Detection')

@st.cache_data
def load_data(nrows):
    df = pd.read_csv(r'C:\ML4B Project\creditcard_2023.csv')
    lowercase = lambda x: str(x).lower()
    df.rename(lowercase, axis='columns', inplace=True)
    return df


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
df = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache_data)")


st.subheader('Raw data')
st.write(df)

st.subheader('Correlation between each column and Class column')


