# Machine Learning Credit Card Fraud Detection

**Authors**: Jeremi Degenhardt (StudOn-Username/Enrollment Number), Leo Gfeller (StudOn-Username/Enrollment Number), Frederic von Gahlen (StudOn-Username/Enrollment Number), Alexander Nigg (StudOn-Username/Enrollment Number)

## Motivation
Our team wants to develop a machine learning program that can automatically detect fraudulent credit card transactions and flag them as such.

## 2 Related Work
There have been a few projects on the area of credit card fraut detection, but nearly always the features have been anonymized. We wanted to firstly display what the given data depicted in an abstract context, visualize the given data and then predict if the transaction is fraudulent or not with different approaches.

## 3 Methodology
### 3.1 General Methodology
Since theres already projects that worked with credit card fraud detection, we looked at what others have already done in that field, and compared that with the approaches we wanted to try out. Then we looked at different machine learnen processes e. g. logistic regression or search trees and compared the performance of the different approaches.

### 3.2 Data Understanding and Preparation
We have choosen a big dataset with nearly 550k entries. The dataset involves different features ranging from V1 to V27. One of our biggest challanges was the anonymization of our dataset which makes it harder to understand what excactly is going on and also what features the algorithm is working with. Then we also have an output feature determening if the transaction was fraudulent or not. Forutnately, the dataset was quite well maintained, had no null values and is also really well distributed (50% fraudulent/non-fraudulent transactions)

### 3.3 Modeling and Evaluation
We have chosen to try out different model architectures to predict our outcome and to compare these different outputs. We chose the logistic regression model, the random decision tree and the k-means clustering to work with and to be compare to one another .

## 4 Results
Describe what artifacts you have built. List the libraries and tools you used. Explain the concept of your app. Describe the results you achieved by applying your trained models on unseen data. Use descriptive language (no judgment, no discussion in this section -> just show what you built).

## 5 Discussion
Now it's time to discuss your results, artifacts, and app. Show the limitations (e.g., missing data, limited training resources/GPU availability in Colab, limitations of the app). Discuss your work from an ethics perspective: Dangers of the application of your work (for example, discrimination through ML models), transparency effects on climate change. Possible sources: Have a look at the "Automating Society Report"; have a look at this website and their publications. Further Research: What could be the next steps for other researchers (specific research questions).

## 6 Conclusion
Short summary of your findings and outlook.
