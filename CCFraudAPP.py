import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("creditcard_2023.csv")

def home():
    st.title("Credit Card Fraud Detection")
    st.subheader("Goal of the Project")
    st.write("Our team took on the task to develop a ML-model which given certain parameters of a credit card transaction could determine wether a transaction was fraudulent or not. ")

    st.subheader("Motivation")
    st.markdown("""According to the 2023 credit card fraud report released by Security.org, that <span style="color: red;">65%</span> of <span style="color: red;">U.S adults</span> have atleast once expirienced a fraudulent transaction on their credit card. While <span style="color: red;">credit card fraud</span> numbers are <span style="color: red;">steady to rising in the U.S.</span>, the European Central Bank actually released statistics that show a <span style="color: red;">decline in creditcard fraud in Europe</span>. Eventhough those are good news for Europeans, the issue of credit card fraud is not going away any time soon. Credit card companies and criminals are in a rat race, where both become more sofisticated in either preventing or conducting credit card fraud. That is why <span style="color: red;">ML-Models</span> are becoming an immensly important tool to <span style="color: red;">recognize and flag fraudulent credit crad transactions</span> as reliably and quickly as possible. """, unsafe_allow_html=True)

    st.title("Logistic Regression")
    
    st.subheader("What is Logistic Regression")
    st.markdown("""Logistic Regression is a statistical method for binary classification. This means it helps us predict one of two possible outcomes. In our case that would be the distiction between fraudulant and non-fraudulant transaction. The model makes those decisions based on variouse factors and features.""", unsafe_allow_html=True)

    st.subheader("How does it work?")
    st.markdown("""<strong>Logistic Function:</strong> While linear regression models predict continuous values, a logistical regressionn model predicts probabilities. The outputs of the regression model are values between 0 and 1. This makes a logistical regression model perfect for binary classification tasks.""", unsafe_allow_html=True)
    st.markdown("""<strong>Probability to Classes:</strong> After we get this probability as an output we set a threshold (usually 0.5) to classify the probablility into two classes. In our instance that would mean that if the probablity that a transaction is fraudulant is greater than 50 percent it would get flagged.""", unsafe_allow_html=True)
    st.markdown("""<strong>Model training:</strong> During the models training it adjusts its parameters to best fit the data. This involves finding the best curve (example of curve below) that seperates the two classes by maximizing the likelihood of the observed data. This training takes place ina supervised enviroment.""", unsafe_allow_html=True)
    with st.expander("Show logistical Regression Curve"):
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
    The columns of our dataset consist of <span style="color: red;">28 anonymized Parameters</span>, 
    the <span style="color: red;">transaction-amount in $</span>, a  
    <span style="color: red;">unique Id</span>, and the <span style="color: red;">Class</span> which determines 
    if a transaction was <span style="color: red;">fraudulent or not</span> (1 = fraudulent; 0 = not fraudulent). 
    The <span style="color: red;">total number of rows</span> in our dataset is <span style="color: red;">568630</span>.
    """, unsafe_allow_html=True)
    with st.expander("Show raw data"):
        st.write(data.shape)
        st.write(data.head())

    st.subheader("Correlation Matrix")
    st.markdown("""The highest correlation between parameters are between <span style="color: red;">V16</span>, <span style="color: red;">V17</span> and <span style="color: red;">V18</span>.""", unsafe_allow_html=True)
    heatmap = plt.figure(figsize=[20,10])
    sns.heatmap(data.corr(),cmap="crest", annot=True)
    with st.expander("Show correlation heatmap"):
        st.pyplot(heatmap)


    st.subheader("Feature Correlations with Class")
    st.markdown("""The features <span style="color: red;">V2</span>, <span style="color: red;">V3</span>, <span style="color: red;">V4</span>, <span style="color: red;">V9</span>, <span style="color: red;">V10</span>, <span style="color: red;">V11</span>, <span style="color: red;">V12</span> and <span style="color: red;">V14</span> seem to have the <span style="color: red;">highest correlation</span> with our classification column. """, unsafe_allow_html=True)
    with st.expander("Show Feature Correlations with Class"):
        data_no_id = data.drop(columns=['id'])
        correlation_matrix = data_no_id.corr()

        # Extracting the correlations with the 'Class' column
        class_correlation = correlation_matrix['Class']

        # Dropping the 'Class' correlation with itself and sorting the values
        class_correlation_sorted = class_correlation.drop('Class').sort_values(ascending=False)

        # Creating the bar plot
        plt.figure(figsize=(10, 6), facecolor='#2e2e2e')
        ax = sns.barplot(x=class_correlation_sorted.values, y=class_correlation_sorted.index, palette='coolwarm')
        
        # Adding title and labels for clarity
        plt.title('Feature Correlations with Class', color='white')
        plt.xlabel('Correlation Coefficient', color='white')
        plt.ylabel('Features', color='white')

        # Customize the tick parameters
        ax.tick_params(colors='white', which='both')  # both major and minor ticks are affected
        plt.setp(ax.get_xticklabels(), color='white')
        plt.setp(ax.get_yticklabels(), color='white')

        # Display the plot
        st.pyplot(plt)
    

    
    st.subheader("Distribution of Parameter Amount and Classifier Class")
    st.markdown("""Our dataset is <span style="color: red;">equally devided</span> into <span style="color: red;">fraudulent</span> and <span style="color: red;">non-fraudulent</span> transactions. The parameter <span style="color: red;">amount</span> meaning the amount of money send in a transaction is <span style="color: red;"> evenly distributed</span>.""", unsafe_allow_html=True)
    with st.expander("Show Visual for Amount or Class"):
        visual = st.selectbox("Select Visual", ["Distribution of Amount", "Distribution of Class"])
        
        if visual == "KDE Plot of Amount":
            plt.figure(figsize=[10, 6])
            sns.kdeplot(data=data["Amount"], fill=True, color="skyblue", bw_adjust=0.5)
            plt.title("KDE Plot of Transaction Amount")
            plt.xlabel("Transaction Amount")
            plt.ylabel("Density")
            plt.grid(True)
            st.pyplot(plt)

        elif visual == "Class Distribution Pie Chart":
            fig, ax = plt.subplots(figsize=[10, 6], facecolor='#2e2e2e')  # Anthracite background
            fig.patch.set_facecolor('#2e2e2e')  # Set the figure background color

            data["Class"].value_counts().plot.pie(
                autopct="%3.1f%%", 
                colors=["lightblue", "red"], 
                startangle=140, 
                explode=(0.1, 0), 
                ax=ax
            )
            plt.title("Distribution of Class", color='white')
            plt.ylabel("")  # Hide the y-label for better visual appeal
            ax.legend(["Not Fraudulent", "Fraudulent"], loc="upper right", framealpha=0.5)
            plt.setp(ax.get_legend().get_texts(), color='white')  # Set legend text color to white
            plt.setp(ax.texts, color='white')  # Set pie chart text color to white
            st.pyplot(fig)


def model_training():

    x = data.drop(['id','Class'],axis=1) # Id not important for our use
    y = data.Class # Save Class attribute (fraudelent/non-fraudulent in own variable)

    sc = StandardScaler()
    x_scaled = sc.fit_transform(x) 
    x_scaled_df = pd.DataFrame(x_scaled,columns=x.columns)

    x_train,x_test,y_train,y_test = train_test_split(x_scaled_df,y,test_size=0.25,random_state=15,stratify= y) 
    # Creating training and testing dataset
    
    st.title("Model Training")
    
    st.subheader("Data Preparation and Preprocessing")

    st.markdown("""
    To start the process of training our ML-Model we need to prepare our dataset. 
    First we <span style="color: red;">divide</span> our dataset into <span style="color: red;">dependent</span> 
    and <span style="color: red;">independent</span> features. In our dataset, the feature 
    <span style="color: red;">Class</span> (fraudulent/non-fraudulent) is what we want our 
    model to <span style="color: red;">predict</span>, that is our <span style="color: red;">dependent</span> feature. 
    <span style="color: red;">All other features</span> are <span style="color: red;">independent</span>. 
    Apart from our feature <span style="color: red;">Class</span>, we will also <span style="color: red;">drop</span> 
    the feature <span style="color: red;">Id</span> since it just is <span style="color: red;">not important</span> 
    for the prediction. In our <span style="color: red;">preprocessing</span> step, we use 
    <span style="color: red;">standardization</span> to bring all our <span style="color: red;">features</span> 
    on the <span style="color: red;">same scale</span>.
    """, unsafe_allow_html=True)

    st.subheader("Building our ML-Model")
    st.markdown("""
    We saw our best chances at building a highly accurate ML-Model in the <span style="color: red;">logistic regression</span> 
    approach. We decided to go with this approach because it is a <span style="color: red;">supervised machine learning algorithm</span>, 
    which is used for <span style="color: red;">binary classification tasks</span> (in our case fraudulent/non-fraudulent) 
    and is <span style="color: red;">perfect for predictive modeling</span>. While we already did our data preparation and preprocessing 
    we still need to <span style="color: red;">divide</span> our dataset into a <span style="color: red;">training-set</span> 
    and a <span style="color: red;">test-set</span>. For this, we went with a <span style="color: red;">3/4</span> to 
    <span style="color: red;">1/4</span> distribution.
    """, unsafe_allow_html=True)


    st.subheader("Our Model is now ready to Run!")
    button_key = "train_model_button_" + str(np.random.randint(1e6))  # Generate a unique key
    if st.button("Train Model", key=button_key):
        lr=LogisticRegression()
        lr.fit(x_train,y_train)

        preds_lr_train = lr.predict(x_train)
        preds_lr_test = lr.predict(x_test)
        
        st.write("Model Accuracy")
        model_eval(y_train, preds_lr_train)

        st.write("Test Accuracy")
        model_eval(y_test, preds_lr_test)



def model_eval(actual, predicted):
    acc_score = accuracy_score(actual, predicted)
    conf_matrix = confusion_matrix(actual, predicted)
    class_rep = classification_report (actual, predicted)
    print("Model Accuracy: ", round(acc_score, 2))
    print(conf_matrix)
    print(class_rep)


# Create a sidebar with navigation options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Training"])

# Display the selected page
if page == "Home":
    home()
elif page == "Data Exploration":
    data_exploration()
elif page == "Model Training":
    model_training()

# Text at the bottom of the sidebar
st.sidebar.markdown(
    """
    <hr style="margin-top: 20px; margin-bottom: 10px;">
    <p style="font-size: 12px;">An App Created by Jeremi Degenhardt, Frederic von Gahlen, Leo Gfeller, and Alexander Nigg</p>
    """,
    unsafe_allow_html=True
)