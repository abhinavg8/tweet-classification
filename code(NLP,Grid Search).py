# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:53:12 2020

@author: Abhinav Gupta
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Train_data.csv') 
Test_set = pd.read_csv('Test_data.csv')
data = dataset.iloc[:,3:6].values.astype(int)
Test_data = Test_set.iloc[:,2:5].values.astype(int)
# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 5365):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 80)
data2= cv.fit_transform(corpus).toarray()
X_train= np.append(data,data2, axis = 1)
X_train = X_train.astype(int)
y_train= dataset.iloc[:, 1].values


#test set 
corpus = []
for i in range(0, 2000):
    review = re.sub('[^a-zA-Z]', ' ', Test_set['Tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
data = Test_set.iloc[:,2:5].values.astype(int)
cv = CountVectorizer(max_features = 80)
data2= cv.fit_transform(corpus).toarray()
X_test= np.append(data,data2, axis = 1)
X_test= X_test.astype(int)


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 4000, num = 40)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 140, num = 14)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 150, cv = 5, 
                               verbose=2, n_jobs = -1)

rf_random.fit(X_train,y_train)


classifier = RandomForestClassifier(n_estimators = 1661,
 min_samples_split = 2,
 min_samples_leaf = 1,
 max_features = 'auto',
 max_depth = None,
 bootstrap = False)

classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

final = pd.DataFrame(y_pred)
Test_set['User'] = final

Test_set.drop(['Tweet','Retweet count','Likes count','Tweet value'], axis = 1,inplace = True)


Test_set.to_csv('Answer.csv',index = False)
