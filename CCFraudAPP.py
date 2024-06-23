from sklearn import model_selection, neighbors
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px

# Load the new dataset
data = pd.read_csv("card_transdata.csv")

def home():
    st.title("Credit Card Fraud Detection")
    st.subheader("Goal of the Project")
    st.write("Our team took on the task to develop a ML-model which given certain parameters of a credit card transaction could determine whether a transaction was fraudulent or not.")

    st.subheader("Motivation")
    st.markdown("""According to the 2023 credit card fraud report released by Security.org, that <span style="color: red;">65%</span> of <span style="color: red;">U.S adults</span> have at least once experienced a fraudulent transaction on their credit card. While <span style="color: red;">credit card fraud</span> numbers are <span style="color: red;">steady to rising in the U.S.</span>, the European Central Bank released statistics that show a <span style="color: red;">decline in credit card fraud in Europe</span>. Even though those are good news for Europeans, the issue of credit card fraud is not going away any time soon. Credit card companies and criminals are in a rat race, where both become more sophisticated in either preventing or conducting credit card fraud. That is why <span style="color: red;">ML-Models</span> are becoming an immensely important tool to <span style="color: red;">recognize and flag fraudulent credit card transactions</span> as reliably and quickly as possible.""", unsafe_allow_html=True)

    st.title("Logistic Regression")
    
    st.subheader("What is Logistic Regression")
    st.markdown("""Logistic Regression is a statistical method for binary classification. This means it helps us predict one of two possible outcomes. In our case, that would be the distinction between fraudulent and non-fraudulent transactions. The model makes those decisions based on various factors and features.""", unsafe_allow_html=True)

    st.subheader("How does it work?")
    st.markdown("""<strong>Logistic Function:</strong> While linear regression models predict continuous values, a logistic regression model predicts probabilities. The outputs of the regression model are values between 0 and 1. This makes a logistic regression model perfect for binary classification tasks.""", unsafe_allow_html=True)
    st.markdown("""<strong>Probability to Classes:</strong> After we get this probability as an output we set a threshold (usually 0.5) to classify the probability into two classes. In our instance, that would mean that if the probability that a transaction is fraudulent is greater than 50 percent, it would get flagged.""", unsafe_allow_html=True)
    st.markdown("""<strong>Model training:</strong> During the model's training, it adjusts its parameters to best fit the data. This involves finding the best curve (example of curve below) that separates the two classes by maximizing the likelihood of the observed data. This training takes place in a supervised environment.""", unsafe_allow_html=True)
    with st.expander("Show Logistic Regression Curve"):
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Exam_pass_logistic_curve.svg/600px-Exam_pass_logistic_curve.svg.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    st.subheader("Why are we using Logistic Regression")
    st.markdown("""<strong>Simplicity:</strong> Logistic regression is simple and easy to implement. It's a good starting point for binary classification problems.""", unsafe_allow_html=True)
    st.markdown("""<strong>Interpretability:</strong> The model is easy to interpret. The coefficients (parameters) can give insights into how each feature impacts the prediction.""", unsafe_allow_html=True)
    st.markdown("""<strong>Efficiency:</strong> It is computationally efficient and works well with small to medium-sized datasets.""", unsafe_allow_html=True)
    st.markdown("""<strong>Performance:</strong> Despite its simplicity, logistic regression often provides good performance and can be a strong baseline for more complex models.""", unsafe_allow_html=True)

def data_exploration():
    st.title("Data Exploration")
    st.subheader("Raw Data")
    st.markdown("""
    The columns of our dataset consist of <span style="color: red;">8 transaction attributes</span> and the <span style="color: red;">Class</span> which determines 
    if a transaction was <span style="color: red;">fraudulent or not</span> (1 = fraudulent; 0 = not fraudulent). 
    The <span style="color: red;">total number of rows</span> in our dataset is <span style="color: red;">100,000</span>.
    """, unsafe_allow_html=True)
    with st.expander("Show raw data"):
        st.write(data.shape)
        st.write(data.head())

    if st.checkbox('Show interactive correlation heatmap'):
        fig = px.imshow(data.corr(), text_auto=True, aspect='auto', color_continuous_scale='viridis')
        st.plotly_chart(fig)
    else:
        heatmap = plt.figure(figsize=[20, 10])
        sns.heatmap(data.corr(), cmap="crest", annot=True)
        with st.expander("Show correlation heatmap"):
            st.pyplot(heatmap)

    st.subheader("Feature Correlations with Class")
    st.markdown("""The features <span style="color: red;">distance_from_home</span>, <span style="color: red;">distance_from_last_transaction</span>, <span style="color: red;">ratio_to_median_purchase_price</span> seem to have the <span style="color: red;">highest correlation</span> with our classification column.""", unsafe_allow_html=True)
    with st.expander("Show Feature Correlations with Class"):
        correlation_matrix = data.corr()
        class_correlation = correlation_matrix['fraud']
        class_correlation_sorted = class_correlation.drop('fraud').sort_values(ascending=False)

        plt.figure(figsize=(10, 6), facecolor='#2e2e2e')
        ax = sns.barplot(x=class_correlation_sorted.values, y=class_correlation_sorted.index, palette='coolwarm')
        
        plt.title('Feature Correlations with Class', color='white')
        plt.xlabel('Correlation Coefficient', color='white')
        plt.ylabel('Features', color='white')

        ax.tick_params(colors='white', which='both')
        plt.setp(ax.get_xticklabels(), color='white')
        plt.setp(ax.get_yticklabels(), color='white')

        st.pyplot(plt)

    st.subheader("Distribution of Class Labels")
    st.markdown("""Our dataset is <span style="color: red;">imbalanced</span> with <span style="color: red;">91.3%</span> non-fraudulent and <span style="color: red;">8.7%</span> fraudulent transactions. We have resampled the dataset to have equal classes.""", unsafe_allow_html=True)

    with st.expander("Show Visual for Amount or Class"):
        visual = st.selectbox("Select Visual", ["Distribution of Fraud", "Distribution of Class"])
    
        if visual == "Distribution of Fraud":
            class_distribution = data["fraud"].value_counts().reset_index()
            class_distribution.columns = ["Class", "Count"]
            fig = px.pie(class_distribution, values="Count", names="Class", title="Distribution of Fraud", color_discrete_sequence=["lightblue", "red"])
            st.plotly_chart(fig)

        elif visual == "Distribution of Class":
            real = data[data["fraud"] == 0]
            fraud = data[data["fraud"] == 1]

            real_resample = real.sample(n=87403, random_state=123)
            fraud_resample = fraud.sample(n=87403, random_state=123)
            data_corrected = pd.concat([real_resample, fraud_resample], axis=0)

            class_distribution_corrected = data_corrected["fraud"].value_counts().reset_index()
            class_distribution_corrected.columns = ["Class", "Count"]
            fig = px.pie(class_distribution_corrected, values="Count", names="Class", title="Distribution of Class (Resampled)", color_discrete_sequence=["lightblue", "red"])
            st.plotly_chart(fig)

def model_training():
    st.title("Model Training")

    st.subheader("Data Preparation and Preprocessing")
    st.markdown("""
    To start the process of training our ML-Model, we need to prepare our dataset. 
    First, we divide our dataset into dependent and independent features. 
    In our dataset, the feature "fraud" (fraudulent/non-fraudulent) is what we want our 
    model to predict; thus, it is our dependent feature. 
    In our preprocessing step, we use standardization to bring all our features to the same scale.
    """)


    # Check if 'fraud' column exists in the dataset
    if 'fraud' not in data.columns:
        st.error("Error: 'fraud' column not found in the dataset. Please ensure the target variable is correctly labeled.")
        st.stop()

    # Prepare data
    x = data.drop(['fraud'], axis=1)  # Features
    y = data['fraud']  # Target variable

    # Standardize features
    sc = StandardScaler()
    x_scaled = sc.fit_transform(x)
    x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_scaled_df, y, test_size=0.25, random_state=15)

    st.subheader("Building our ML-Model")

    st.markdown("""
    We believe logistic regression offers a promising approach for this problem due to its simplicity, interpretability, and efficiency in binary classification tasks.
    We'll divide our dataset into a training set and a test set with a 3:1 ratio.
    """)

    # Model training and evaluation
    train_and_evaluate_model("Logistic Regression", LogisticRegression(max_iter=1000), x_train, y_train, x_test, y_test, x)
    train_and_evaluate_model("Decision Tree", DecisionTreeClassifier(random_state=0), x_train, y_train, x_test, y_test, x)
    train_and_evaluate_model("Random Forest", RandomForestClassifier(n_estimators=50, random_state=0), x_train, y_train, x_test, y_test, x)

def train_and_evaluate_model(model_name, model, x_train, y_train, x_test, y_test, x):
    st.subheader(f"Train {model_name} Model")
    if st.button(f'Train {model_name} Model'):
        model.fit(x_train, y_train)

        # Predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Evaluation metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        train_conf_matrix = confusion_matrix(y_train, y_train_pred)
        test_conf_matrix = confusion_matrix(y_test, y_test_pred)

        train_class_report = classification_report(y_train, y_train_pred, output_dict=True)
        test_class_report = classification_report(y_test, y_test_pred, output_dict=True)

        # Display evaluation results
        st.balloons()
        st.markdown(f"<h2 style='text-align: center;'>{model_name} Model Evaluation</h2>", unsafe_allow_html=True)

        st.markdown("<h3>Training Accuracy</h3>", unsafe_allow_html=True)
        st.write(f"<div style='text-align: center; font-size: 24px; color: green;'>{train_accuracy:.2f}</div>", unsafe_allow_html=True)

        st.markdown("<h3>Confusion Matrix</h3>", unsafe_allow_html=True)
        st.write("Training Confusion Matrix:")
        st.dataframe(pd.DataFrame(train_conf_matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(train_class_report).transpose())

        st.markdown("<h3>Test Accuracy</h3>", unsafe_allow_html=True)
        st.write(f"<div style='text-align: center; font-size: 24px; color: green;'>{test_accuracy:.2f}</div>", unsafe_allow_html=True)

        st.markdown("<h3>Confusion Matrix</h3>", unsafe_allow_html=True)
        st.write("Test Confusion Matrix:")
        st.dataframe(pd.DataFrame(test_conf_matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(test_class_report).transpose())

        # Visualizations
        if model_name in ["Logistic Regression", "Decision Tree"]:
            st.markdown(f"<h3>{model_name} Model Visualization</h3>", unsafe_allow_html=True)

        # ROC Curve and AUC
        if model_name == "Logistic Regression":
            from sklearn.metrics import roc_curve, auc

            # Calculate ROC curve and AUC for Logistic Regression
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title(f'ROC Curve - {model_name}')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

        if model_name == "Decision Tree":
            st.subheader(f"{model_name} Visualization")
            plt.figure(figsize=(15, 10))
            plot_tree(model, filled=True, feature_names=x_train.columns, class_names=["Not Fraud", "Fraud"])
            st.pyplot(plt)

    

def fraud_detector():
    st.header("Fraud Detector")
    st.markdown("""Here you can use our <span style="color: red;">fraud detection tool</span> to check if a transaction was fraudulent or not. Just input the transaction information acording to our form and our <span style="color: red;">k-Means-Clustering</span>-Model will do the rest! """, unsafe_allow_html=True)

    # Create form for user input
    with st.form(key='fraud_detection_form'):
         distance_from_home = st.number_input('Distance from Home in kilometers', min_value=0.0, format='%f')
         distance_from_last_transaction = st.number_input('Distance from Last Transaction in kilometers', min_value=0.0, format='%f')
         ratio_to_median_purchase_price = st.number_input('Ratio to Median Purchase Price', min_value=0.0, format='%f')
    
         repeat_retailer = st.selectbox('Is this purchase from a repeat retailer?', ['YES', 'NO'])
         used_chip = st.selectbox('Was a chip used in this transaction', ['YES', 'NO'])
         used_pin_number = st.selectbox('Was a PIN number used in this transaction?', ['YES', 'NO'])
         online_order = st.selectbox('Is this an online order', ['YES', 'NO'])
    
         submit_button = st.form_submit_button(label='Check your Transaction')

    # Handle form submission
    if submit_button:

        # Map YES/NO to 1/0
        repeat_retailer = 1 if repeat_retailer == 'YES' else 0
        used_chip = 1 if used_chip == 'YES' else 0
        used_pin_number = 1 if used_pin_number == 'YES' else 0
        online_order = 1 if online_order == 'YES' else 0

        input_data = np.array([[distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price, 
                                repeat_retailer, used_chip, used_pin_number, online_order]])
        
        prediction = fraud_detector_model(input_data)
        color = "red" if prediction == "Fraudulent" else "green"
        st.markdown(f"<h3 style='color: {color};'>{prediction}</h3>", unsafe_allow_html=True)

def fraud_detector_model(input_data):
    real = data[data["fraud"] == 0]
    fraud = data[data["fraud"] == 1]

    # Resampling the original dataset with 87,403 datapoints for both classes
    real_resample = real.sample(n=87403, random_state=123)
    fraud_resample = fraud.sample(n=87403, random_state=123)

    # Creating new dataset consisting of equal class occurrence
    data_corrected = pd.concat([real_resample, fraud_resample], axis=0)

    x = np.array(data_corrected.drop(columns="fraud"))
    y = np.array(data_corrected["fraud"])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2, random_state=123, shuffle=True)

    # Data scaling to produce good results
    scale = MinMaxScaler()
    x_train = scale.fit_transform(x_train)
    x_test = scale.transform(x_test)

    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)

    input_data_scaled = scale.transform(input_data)
    prediction = knn.predict(input_data_scaled)
    return "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"

# Create a sidebar with navigation options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Training", "Fraud Detector"])

# Display the selected page
if page == "Home":
    home()
elif page == "Data Exploration":
    data_exploration()
elif page == "Model Training":
    model_training()
elif page == "Fraud Detector" :
    fraud_detector()  

# Text at the bottom of the sidebar
st.sidebar.markdown(
    """
    <hr style="margin-top: 20px; margin-bottom: 10px;">
    <p style="font-size: 12px;">An App Created by Jeremi Degenhardt, Frederic von Gahlen, Leo Gfeller and Alexander Nigg</p>
    """,
    unsafe_allow_html=True
)
