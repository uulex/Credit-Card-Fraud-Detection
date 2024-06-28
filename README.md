# Machine Learning Credit Card Fraud Detection

**Authors**: Jeremi Degenhardt (ez03obyk/23121559), Leo Gfeller (aj20ibuj/23188883), Frederic von Gahlen (ge29zogy/23156365), Alexander Nigg (to98wiju/23220114)

![this is how credit card fraud works (believe us we are experts)](https://github.com/uulex/ML4B/blob/main/dataset-cover.jpg?raw=true)

[Streamlit App](https://ccfraud.streamlit.app)

## Motivation
Our team wants to develop a machine learning program that can automatically detect fraudulent credit card transactions and flag them as such.

## 2 Related Work
There have been a few projects on the area of credit card fraut detection, but nearly always the features have been anonymized. We wanted to firstly display what the given data depicted in an abstract context, visualize the given data and then predict if the transaction is fraudulent or not with different approaches.

## 3 Methodology
### 3.1 General Methodology
Since theres already projects that worked with credit card fraud detection, we looked at what others have already done in that field, and compared that with the approaches we wanted to try out. Then we looked at different machine learnen processes e. g. logistic regression or search trees and compared the performance of the different approaches.

### 3.2 Data Understanding and Preparation
We have choosen a big dataset with 1000k entries. The dataset involves 8 different features like distance from home or if a pin code was used in the transaction. We also have an output feature determening if the transaction was fraudulent or not. Forutnately, the dataset was quite well maintained, had no null values and was quite easy to work with. We had to perform some down sampling, since our distribution of fraudulent/non-fraudulent transactions was highly skewed. After we resampled the dataset, we achived a 50/50 distribution, while still working with 175k entries.

### 3.3 Modeling and Evaluation
We have chosen to try out different model architectures to predict our outcome and to compare these different outputs. We chose the logistic regression model, the random decision tree and the k-means clustering to work with and to be compare to one another .

## 4 Results

Our app firstly explains to the user, what models we have used, explaining what the different models do and how they work. Then the app presents our used dataset and guides the user through the process of preprocessing. Then the user can see how the different models performed on our dataset.
After comparing the different ML-Models, the app offers the user an interactive interface to input a credit card transaction. The trained model will then predict wether the transaction was fraudulent or non-fraudulent.

## 5 Discussion
We are really happy with the results of our app. It predicts accurately and consistent. One downside might be the false flagging of transactions, when the credit card holder is for example traveling. Then the distance between last transaction and also distance from home is drasticly changing, resulting in a false flagging (which actually happens in the real world).  You also have to keep in mind, in order to use this model in the real world, it would have to undergo a lot of testing, since a small amount of false negatives could have a huge impact on the credit card holders. Therefore the next step could be to improve the model even further and make it usable for real world purposes.

## 6 Conclusion
After finding a suitable dataset, we preprocessed the given data, in the process reducing the number of entries but also making the data more evenly distribuited. We then compared the three different approaches and saw that random search forest gave us the best results. We then developed an interactive interface for the user of the app to enter a credit card transaction and to check wheter it is flagged as fraudulent or non-fraudulent by our model.

