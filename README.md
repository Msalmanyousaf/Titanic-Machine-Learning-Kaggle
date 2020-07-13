# Titanic: Machine Learning from Disaster

This is a very well-known Kaggle project which gives a nice introduction to data science/machine learning. The dataset contains information about the passengers of Titanic ship, which sank in the North Atlantic Ocean in 1912. 

After carefully examining the dataset and learning from various resources shared publicly on Kaggle platform, machine learning classification concepts were applied. The project provided a hands-on experience of data pre-processing (data cleaning, feature engineering) and different classification algorithms.
## 1. Input Features
**PassengerId:** Integer showing id of the passengers  
**Pclass:** Ticket class (1, 2 or 3).  
**Name:** Passenger name  
**Sex:** male/female  
**Age:** Passeneger age in years  
**SibSp:** No. of siblings/spouses aboard the ship  
**Parch:** No. of parents/childeren aboard the ship  
**Ticket:** Ticket number  
**Fare:** Passenger fare  
**Cabin:** Passenger cabin number  
**Embarked:** Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
## 2. Output
**Survived:** Target feature. 1 if the passenger survived and 0 otherwise.  
## 3. Machine Learning Problem
Given the input feature values, the task is to train a classifier which predicts whether a particular passenger survived (1) or not (0).
### 3.1.  Files
**train.csv:** File containing training data.
**test.csv:** File containing the input feature values of the passengers for whom the predictions are required.
### 3.2. Approach
* A thorough data pre-processing is performed. All the steps can be found in the notebook file "Titanic_Data_preprocessing.ipynb".
* 3 classification models are tried:
-- Random Forest Classifier
-- Support Vector Machine Classifier
-- Gradient Boosting Classifier
	Moreover, grid search is used to tune the parameters of each model and finally, VotingClassifier is used to get the ensemble model.
* Stratified k-fold cross validation is used with 10 splits. The final model showed the accuracy score of around 0.83.
