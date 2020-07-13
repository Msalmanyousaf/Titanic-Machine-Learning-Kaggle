# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 21:34:14 2020

@author: Salman
"""
import numpy as np
import pandas as pd


def data_preprocessing (data_frame, train_size):
    """
    The function takes a dataframe as an input, does the data_preprocessing 
    (missing values handling, feature engineering) and returns trainng and test
    data which can be fed to a machine learning model. For detailed understanding
    of the data preprocessing, please refer to the notebook 
    Titanic_Data_preprocessing.ipynb

    Parameters
    ----------
    data_frame : pandas dataframe
    train_size : int
        Number of observations in the training data.
    

    Returns
    -------
    train : pandas dataframe
        It contains the input feature data for training 
    train_y : pandas dataframe
        Target values for training
    test : pandas dataframe
        unseen data for which the predictions are required to be made. it does 
        not have the target values.

    """
    ##################### Handling missing values #######################
    cols_with_miss = [col for col in data_frame.columns 
                      if data_frame[col].isnull().any()]
    # missing value columns: 'Age', 'Fare', 'Cabin', 'Embarked'
    
    # treating the only missing value in 'Fare'
    data_frame.Fare.fillna(data_frame['Fare'].median(), inplace = True)
    
    # in 'Embarked', most frequent value is 'S', so fill the missing with it
    data_frame.Embarked.fillna('S', inplace = True)  
    
    # handling missing values in 'Age" column based on values in the columns
    # of 'Pclass', 'SibSp' and 'Parch'
    Age_nan_indices = list(data_frame['Age'][data_frame['Age'].isnull()].index)
    for i in Age_nan_indices:
        pred_Age = data_frame['Age'][ ((data_frame['Pclass'] == data_frame.loc[i]['Pclass']) 
                                        & (data_frame['SibSp'] == data_frame.loc[i]['SibSp'])
                                        & (data_frame['Parch'] == data_frame.loc[i]['Parch']))].median()
        if  not np.isnan(pred_Age):
            data_frame['Age'].loc[i] = pred_Age
        else:
            data_frame['Age'].loc[i] = data_frame['Age'][ (data_frame['SibSp'] == data_frame['SibSp'].loc[i])].mean()     
    
    ###################### Feature Engineering ###############################   
    
    # 'Name' feature: capture the Titles from the names of each passenger
    names = [naam.split(',')[1].split('.')[0].strip() for naam in list(data_frame.Name)]
    
    data_frame.drop("Name", axis = 1, inplace = True)
    data_frame["Title"] = names
    
    # Convert to categorical values Title 
    data_frame["Title"] = data_frame["Title"].replace(['Lady', 'the Countess',
                                                       'Countess', "Mlle",
                                              'Capt', 'Col','Don', 'Dr', 'Major', 
                                              'Rev', 'Sir', 'Jonkheer', 'Dona',
                                              "Ms", "Mme"], 'Rare')
    data_frame["Title"] = data_frame["Title"].map({"Master":0, "Miss":1, 
                                          "Mrs":2, "Mr":3, "Rare":4})
    data_frame["Title"] = data_frame["Title"].astype(int)
    
    # converting 'Sex' feature into labeled data
    data_frame.Sex = data_frame.Sex.map({'male':0, 'female':1})
    
    # drop the 'Ticket' column
    data_frame.drop('Ticket', axis = 1, inplace = True)
    
    # get dummies of 'Embarked' variable
    data_frame = pd.get_dummies(data_frame, columns = ['Embarked'], prefix = 'Em_')
    
    # 'Cabin' feature: The missing values may be because of the reason that 
    # some passengers did not have any cabin. so assign them a cabin of 'X'
    data_frame.Cabin.fillna('X', inplace = True)
    # get the first letter of the Cabin feature values and use it as the Cabin
    # categories.
    cabins = [cab[0] for cab in list(data_frame.Cabin)]
    data_frame.Cabin = cabins
    
    # converting into dummies
    data_frame = pd.get_dummies(data_frame, columns = ['Cabin'], prefix = 'Cab_')
    
    # Drop 'PassengerId' column
    data_frame.drop("PassengerId", axis = 1, inplace = True)
    
    
    train = data_frame[:train_size]   
    test = data_frame[train_size:]
    test.drop('Survived', axis = 1, inplace = True)
    train_y = train.Survived
    train.drop('Survived', axis = 1, inplace = True)
    
    return train, train_y, test
