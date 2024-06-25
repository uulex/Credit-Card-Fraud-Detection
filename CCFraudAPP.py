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
from PIL import Image, ImageDraw, ImageFont

# Load the new dataset
data = pd.read_csv("card_transdata.csv")

def home():
    st.title("Credit Card Fraud Detection")
    st.subheader("Goal of the Project")
    st.write("Our team took on the task to develop a ML-model which given certain parameters of a credit card transaction could determine whether a transaction was fraudulent or not.")
    
    st.subheader("Motivation")
    st.markdown("""According to the 2023 credit card fraud report released by Security.org, that <span style="color: red;">60%</span> of <span style="color: red;">U.S credit card holders</span> have at least once experienced a fraudulent transaction on their credit card. And <span style="color: red;">credit card fraud</span> numbers are <span style="color: red;">rapidly rising in the U.S.</span>. That's why the issue of credit card fraud is getting more and more important. Credit card companies and criminals are in a rat race, where both become more sophisticated in either preventing or conducting credit card fraud. That is why <span style="color: red;">ML-Models</span> are becoming an immensely important tool to <span style="color: red;">recognize and flag fraudulent credit card transactions</span> as reliably and quickly as possible.""", unsafe_allow_html=True)

    st.write("We want to give you a quick overview over the three models we used to predict credit card fraud before jumping into our dataset.")
    st.title("Logistic Regression")
    
    st.subheader("What is Logistic Regression")
    st.markdown("""Logistic Regression is a statistical method for binary classification. This means it helps us predict one of two possible outcomes. In our case, that would be the distinction between fraudulent and non-fraudulent transactions. The model makes those decisions based on various factors and features.""", unsafe_allow_html=True)

    st.subheader("How does it work?")
    st.markdown("""<strong>Logistic Function:</strong> While linear regression models predict continuous values, a logistic regression model predicts probabilities. The outputs of the regression model are values between 0 and 1. This makes a logistic regression model suitable for binary classification tasks.""", unsafe_allow_html=True)
    st.markdown("""<strong>Probability to Classes:</strong> After we get this probability as an output we set a threshold (usually 0.5) to classify the probability into two classes. In our instance, that would mean that if the probability that a transaction is fraudulent is greater than 50 percent, it would get flagged.""", unsafe_allow_html=True)
    st.markdown("""<strong>Model training:</strong> During the model's training, it adjusts its parameters to best fit the data. This involves finding the best curve (example of curve below) that separates the two classes by maximizing the likelihood of the observed data. This training takes place in a supervised environment.""", unsafe_allow_html=True)
    with st.expander("Show Logistic Regression Curve"):
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Exam_pass_logistic_curve.svg/600px-Exam_pass_logistic_curve.svg.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    #st.subheader("Why are we using Logistic Regression")
    #st.markdown("""<strong>Simplicity:</strong> Logistic regression is simple and easy to implement. It's a good starting point for binary classification problems.""", unsafe_allow_html=True)
    #st.markdown("""<strong>Interpretability:</strong> The model is easy to interpret. The coefficients (parameters) can give insights into how each feature impacts the prediction.""", unsafe_allow_html=True)
    #st.markdown("""<strong>Efficiency:</strong> It is computationally efficient and works well with small to medium-sized datasets.""", unsafe_allow_html=True)
    #st.markdown("""<strong>Performance:</strong> Despite its simplicity, logistic regression often provides good performance and can be a strong baseline for more complex models.""", unsafe_allow_html=True)

    st.title("K Means Clustering")

    st.subheader("What is K-means Clustering")
    st.markdown("""K-means Clustering is a type of unsupervised learning method used for clustering. This means it helps us group similar data points together based on their features. In our case, that could be grouping customers based on their purchasing behavior. The model makes those decisions based on various factors and features.""", unsafe_allow_html=True)

    st.subheader("How does it work?")
    st.markdown("""<strong>Initialization:</strong> The algorithm starts by randomly initializing 'k' cluster centers. For our case, we have chosen k = 3, since that gave us the best results. 'k' is a parameter that needs to be specified beforehand.""", unsafe_allow_html=True)
    st.markdown("""<strong>Assignment:</strong> Each data point is assigned to the nearest cluster center. The distance is usually calculated using Euclidean distance.""", unsafe_allow_html=True)
    st.markdown("""<strong>Update:</strong> The cluster centers are recalculated as the mean of all data points belonging to that cluster. This process is repeated until the cluster assignments do not change or a maximum number of iterations is reached.""", unsafe_allow_html=True)
    with st.expander("Show K-means Clustering"):
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/K-means_convergence.gif/440px-K-means_convergence.gif", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    #st.subheader("Why are we using K-means Clustering")
    #st.markdown("""<strong>Simplicity:</strong> K-means is simple and easy to implement. It's a good starting point for clustering problems.""", unsafe_allow_html=True)
    #st.markdown("""<strong>Efficiency:</strong> It is computationally efficient and works well with large datasets.""", unsafe_allow_html=True)
    #st.markdown("""<strong>Interpretability:</strong> The model is easy to interpret. The clusters can give insights into how the data is structured.""", unsafe_allow_html=True)
    #st.markdown("""<strong>Performance:</strong> Despite its simplicity, K-means often provides good performance and can be a strong baseline for more complex models.""", unsafe_allow_html=True)

    st.title("Random Forest")

    st.subheader("What is Random Forest")
    st.markdown("""Random Forest is a type of supervised learning method used for both regression and classification tasks. This means it helps us predict a continuous value or a categorical class based on various factors and features.""", unsafe_allow_html=True)

    st.subheader("How does it work?")
    st.markdown("""<strong>Tree Construction:</strong> The algorithm starts by splitting the data based on a feature that provides the best split, according to a mathematical 'impurity' criterion (like Gini impurity or information gain). This process is repeated recursively, resulting in a tree-like model of decisions.""", unsafe_allow_html=True)
    st.markdown("""<strong>Randomness:</strong> In Random Forests, some randomness is introduced in the selection of the feature to split on, adding extra diversity and robustness to the model.""", unsafe_allow_html=True)
    st.markdown("""<strong>Ensemble:</strong> Typically, a set of different trees is created (a 'forest'). Each tree gives a prediction, and the final prediction is decided by majority vote (for classification) or averaging (for regression).""", unsafe_allow_html=True)
    with st.expander("Show Random Forest"):
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/4e/Random_forest_explain.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    #st.subheader("Why are we using Random Decision Trees")
    #st.markdown("""<strong>Accuracy:</strong> Random Decision Trees usually provide a high accuracy, as they learn complex decision boundaries.""", unsafe_allow_html=True)
    #st.markdown("""<strong>Robustness:</strong> They are robust to outliers and non-linear data.""", unsafe_allow_html=True)
    #st.markdown("""<strong>Interpretability:</strong> Each decision in the tree has a clear interpretation, making the model relatively easy to understand.""", unsafe_allow_html=True)
    #st.markdown("""<strong>Efficiency:</strong> They are computationally efficient and work well with large datasets.""", unsafe_allow_html=True)


def data_exploration():
    st.title("Data Exploration")
    st.subheader("Raw Data")
    st.markdown("""
    The columns of our dataset consist of <span style="color: red;">8 transaction attributes</span> and the <span style="color: red;">Class</span> which determines 
    if a transaction was <span style="color: red;">fraudulent or not</span> (1 = fraudulent; 0 = not fraudulent). 
    The <span style="color: red;">total number of rows</span> in our dataset is <span style="color: red;">1000000</span>.
    """, unsafe_allow_html=True)
    with st.expander("Show raw data"):
        st.write(data.shape)
        st.write(data.head())
    
    with st.expander("Show correlation heatmap"):
          fig = px.imshow(data.corr(), text_auto=True, aspect='auto', color_continuous_scale='viridis')
          st.plotly_chart(fig)


    st.subheader("Feature Correlations with Fraud/Not-Fraud")
    st.markdown("""The features <span style="color: red;">distance_from_home</span>, <span style="color: red;">distance_from_last_transaction</span>, <span style="color: red;">ratio_to_median_purchase_price</span> seem to have the <span style="color: red;">highest correlation</span> with our classification column.""", unsafe_allow_html=True)
    with st.expander("Show Feature Correlations with Fraud/Not-Fraud"):
        correlation_matrix = data.corr()
        class_correlation = correlation_matrix['fraud']
        class_correlation_sorted = class_correlation.drop('fraud').sort_values(ascending=False)

        plt.figure(figsize=(10, 6), facecolor='#2e2e2e')
        ax = sns.barplot(x=class_correlation_sorted.values, y=class_correlation_sorted.index, palette='coolwarm')
        
        plt.title('Feature Correlations with Fraud/Not-Fraud', color='white')
        plt.xlabel('Correlation Coefficient', color='white')
        plt.ylabel('Features', color='white')

        ax.tick_params(colors='white', which='both')
        plt.setp(ax.get_xticklabels(), color='white')
        plt.setp(ax.get_yticklabels(), color='white')

        st.pyplot(plt)

    st.subheader("Distribution of Class Labels")
    st.markdown("""Our dataset is <span style="color: red;">imbalanced</span> with <span style="color: red;">91.3%</span> non-fraudulent and <span style="color: red;">8.7%</span> fraudulent transactions. We have resampled the dataset to have equal classes.""", unsafe_allow_html=True)

    with st.expander("Show Visual for Class Distribution"):
        visual = st.selectbox("Select Visual", ["Distribution of Fraud(original)", "Distribution of Fraud (resampled)"])
    
        if visual == "Distribution of Fraud(original)":
            class_distribution = data["fraud"].value_counts().reset_index()
            class_distribution.columns = ["Class", "Count"]
            fig = px.pie(class_distribution, values="Count", names="Class", title="Distribution of Fraud", color_discrete_sequence=["lightblue", "red"])
            st.plotly_chart(fig)

        elif visual == "Distribution of Fraud (resampled)":
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
    st.header("Model Training")
    st.subheader("Data Preparation")
    st.markdown("""Before we start training and testing our three ML-Models we first need to prepare our data. If you looked at our Data Exploration site you probably saw that our classifying feature <span style="color: red;">fraudulent/not fraudulent</span> is <span style="color: red;">not balanced</span> out of the box, so we had to <span style="color: red;">resample</span> the dataset to <span style="color: red;">equalize this feature</span>. Since the majortiy of the transactions in our dataset are flagged as not fraudulent we had to remove a lot of those entries to balance it.""", unsafe_allow_html=True)
    
    # # Create an image with a matching dark background
    # img_width, img_height = 800, 300
    # page_bg_color = (11, 12, 16)  # Match this to the dark theme background color
    # img = Image.new('RGB', (img_width, img_height), color=page_bg_color)
    # draw = ImageDraw.Draw(img)

    # # Define text properties
    # try:
    #     font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=28)
    # except IOError:
    #     font = ImageFont.load_default(size = 24)

    # # Draw rectangles and text for "Before" and "After"
    # draw.rectangle([50, 100, 250, 200], outline="white", width=4)
    # draw.rectangle([550, 100, 750, 200], outline="white", width=4)

    # # Center text within the rectangles
    # before_text = "Before"
    # before_rows_text = "1,000,000 rows"
    # after_text = "After"
    # after_rows_text = "174,806 rows"

    # # Calculate text widths and positions to center them using textbbox
    # before_text_bbox = draw.textbbox((0, 0), before_text, font=font)
    # before_text_width = before_text_bbox[2] - before_text_bbox[0]
    # before_rows_text_bbox = draw.textbbox((0, 0), before_rows_text, font=font)
    # before_rows_text_width = before_rows_text_bbox[2] - before_rows_text_bbox[0]
    
    # after_text_bbox = draw.textbbox((0, 0), after_text, font=font)
    # after_text_width = after_text_bbox[2] - after_text_bbox[0]
    # after_rows_text_bbox = draw.textbbox((0, 0), after_rows_text, font=font)
    # after_rows_text_width = after_rows_text_bbox[2] - after_rows_text_bbox[0]

    # draw.text(((250 - before_text_width) / 2 + 25, 110), before_text, fill="white", font=font)
    # draw.text(((250 - before_rows_text_width) / 2 + 25, 150), before_rows_text, fill="white", font=font)

    # draw.text(((200 - after_text_width) / 2 + 550, 110), after_text, fill="white", font=font)
    # draw.text(((200 - after_rows_text_width) / 2 + 550, 150), after_rows_text, fill="white", font=font)

    # # Draw an arrow with the text "Resampling" in the middle
    # arrow_text = "Resampling"
    # arrow_text_bbox = draw.textbbox((0, 0), arrow_text, font=font)
    # arrow_text_width = arrow_text_bbox[2] - arrow_text_bbox[0]
    # draw.line([300, 150, 500, 150], fill="white", width=3)
    # draw.polygon([490, 140, 510, 150, 490, 160], fill="white")  # Arrowhead
    # draw.text(((img_width - arrow_text_width) / 2, 110), arrow_text, fill="white", font=font)

    
    # st.image(img, caption='Dataset Size Before and After Resampling')

    #st.image("pictures/LR_Matrix.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    st.markdown("""We are now left with about <span style="color: red;">17.5% of the rows in our dataset</span>, which should still be enough to get well-trained ML-Models. Now that our dataset is balanced we will <span style="color: red;">seperate dependent</span> and <span style="color: red;">independet features</span>, which in our case means <span style="color: red;">droping the classifier</span> and saving it in a <span style="color: red;">seperate array</span>.""", unsafe_allow_html=True)
    st.markdown("""The last step before we can finally start training our models is to <span style="color: red;">seperate</span> our data into <span style="color: red;">training- and testing-data</span>. We decided to go for a <span style="color: red;">75% training</span> and <span style="color: red;">25% testing</span> split.""", unsafe_allow_html=True)

    st.subheader("Logistic Regression")
    st.markdown("""Logistic Regression is a <span style="color: red;">supervised machine learning algorithm</span> used for <span style="color: red;">binary classification</span>. It is well suited for <span style="color: red;">predictive modeling</span>.""", unsafe_allow_html=True)
    st.markdown("""**Let's see how it performs:**""", unsafe_allow_html=True)



    # Data
    data = {
        "Class": ["0.0", "1.0", "macro avg", "weighted avg"],
        "Precision": [0.93, 0.91, 0.92, 0.92],
        "Recall": [0.90, 0.93, 0.92, 0.92],
        "F1-Score": [0.92, 0.92, 0.92, 0.92],
        "Support": [17502, 17460, 34962, 34962]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Style the DataFrame
    styled_df = df.style.set_table_styles(
        [
            {'selector': 'thead th', 'props': [('background-color', '#4f4f4f'), ('color', 'white'), ('font-size', '14px')]},
            {'selector': 'tbody td', 'props': [('background-color', '#f9f9f9'), ('color', '#000000'), ('font-size', '14px')]},
            {'selector': 'th.col_heading.level0', 'props': [('display', 'none')]},  # Hide left column header
            {'selector': 'td.col0', 'props': [('display', 'none')]},  # Hide left column cells
            {'selector': 'thead', 'props': [('border-bottom', '2px solid #4f4f4f')]},
            {'selector': 'tbody tr', 'props': [('border-bottom', '1px solid #dddddd')]},
        ]
    ).set_caption("LR Classification Report")

    # Streamlit app
    st.write("**LR Accuracy:** 0.9194531048883041")  # Bold text for LR Accuracy
    st.write("**LR Validation Accuracy:** 0.9171100051484469")  # Bold text for LR Accuracy

    st.image("pictures/LR_Matrix.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    # Display styled DataFrame without left column
    st.dataframe(styled_df, height=200, width=700)

    st.subheader("K-Means-Clustering")
    st.markdown("""K-Means-Clustering is a <span style="color: red;">unsupervised machine learning algorithm</span> and is used for clustering. It analyzes the realationship between different features and <span style="color: red;">groups similar datapoints</span>. It is good for <span style="color: red;">explorative data analysis and pattern recognition</span>.""", unsafe_allow_html=True)
    st.markdown("""**Let's see how it performs:**""", unsafe_allow_html=True)

    data = {
    "Class": ["0.0", "1.0", "macro avg", "weighted avg"],
    "Precision": [1.00, 1.00, 1.00, 1.00],
    "Recall": [1.00, 1.00, 1.00, 1.00],
    "F1-Score": [1.00, 1.00, 1.00, 1.00],
    "Support": [17502, 17460, 34962, 34962]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Style the DataFrame
    styled_df = df.style.set_table_styles(
        [
            {'selector': 'thead th', 'props': [('background-color', '#4f4f4f'), ('color', 'white'), ('font-size', '14px')]},
            {'selector': 'tbody td', 'props': [('background-color', '#f9f9f9'), ('color', '#000000'), ('font-size', '14px')]},
            {'selector': 'th.col_heading.level0', 'props': [('display', 'none')]},
            {'selector': 'thead', 'props': [('border-bottom', '2px solid #4f4f4f')]},
            {'selector': 'tbody tr', 'props': [('border-bottom', '1px solid #dddddd')]},
        ]
    ).set_caption("KNN Classification Report")

    # Streamlit app
    st.write("**KNN Accuracy:** 0.998498326706902")  # Display KNN Accuracy in subheader style
    st.write("**KNN Validation Accuracy:** 0.9971969566958412")  # Display KNN Accuracy in subheader style
    

    st.image("pictures/K_Means_Matrix.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


    # Display styled DataFrame
    st.dataframe(styled_df, height=200, width=700)


    st.subheader("Random Forest")
    st.markdown("""Random Forest is a <span style="color: red;">supervised machine learning algorithm</span> and is used for <span style="color: red;">classification and regression</span>. It constructs <span style="color: red;">multiple decision trees</span> and <span style="color: red;">aggregates their results</span>. It is good for <span style="color: red;">handling overfitting</span> and <span style="color: red;">improving prediction accuracy</span>.""", unsafe_allow_html=True)
    st.markdown("""**Let's see how it performs:**""", unsafe_allow_html=True)

    data = {
    "Class": ["0.0", "1.0", "macro avg", "weighted avg"],
    "Precision": [1.00, 1.00, 1.00, 1.00],
    "Recall": [1.00, 1.00, 1.00, 1.00],
    "F1-Score": [1.00, 1.00, 1.00, 1.00],
    "Support": [17502, 17460, 34962, 34962]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Style the DataFrame
    styled_df = df.style.set_table_styles(
        [
            {'selector': 'thead th', 'props': [('background-color', '#4f4f4f'), ('color', 'white'), ('font-size', '14px')]},
            {'selector': 'tbody td', 'props': [('background-color', '#f9f9f9'), ('color', '#000000'), ('font-size', '14px')]},
            {'selector': 'th.col_heading.level0', 'props': [('display', 'none')]},
            {'selector': 'thead', 'props': [('border-bottom', '2px solid #4f4f4f')]},
            {'selector': 'tbody tr', 'props': [('border-bottom', '1px solid #dddddd')]},
        ]
    ).set_caption("RF Classification Report")  # Adjusted caption for RF

    # Streamlit app
    st.write("**RF Accuracy:** 0.999942795034609")  # Display RF Accuracy in subheader style
    st.write("**RF Validation Accuracy:** 0.9999141925519135")  # Display KNN Accuracy in subheader style

    st.image("pictures/Random_Forest_Matrix.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


    # Display styled DataFrame
    st.dataframe(styled_df, height=200, width=700)


def fraud_detector():
    st.header("Fraud Detector")
    st.markdown("""Here you can use our <span style="color: red;">fraud detection tool</span> to check if a transaction was fraudulent or not. Just input the transaction information acording to our form and our <span style="color: red;">Random Forest Model</span>-Model will do the rest! """, unsafe_allow_html=True)

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

    rf = RandomForestClassifier(random_state=123)
    rf.fit(x_train, y_train)

    input_data_scaled = scale.transform(input_data)
    prediction = rf.predict(input_data_scaled)
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
