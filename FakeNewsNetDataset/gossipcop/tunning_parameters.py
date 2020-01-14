import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X_train = pd.read_csv("train_gossipcop.csv", ",", usecols=['text', 'label'])
y_train = X_train['label'].values.flatten()
X_train = X_train['text'].values.flatten()
print("Read")

stopwords = set(ENGLISH_STOP_WORDS)

#SVM parameters
# parameters = [  
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]

#2nd round of tuning
# parameters = [  
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma' : [0.001, 0.01, 0.1, 1], 'kernel': ['rbf']}
  
#  ]

#KNN parameters
# parameters = {
#     'n_neighbors' : [3, 5, 11, 19],
#     'weights' : ['uniform', 'distance'],
#     'metric' : ['euclidean', 'manhattan']
# }


#Logistic Regression parameters
# parameters = [
#     {'C': [1, 10, 100, 1000], 'penalty' : ['l2'], 'solver' : ['sag', 'saga'] },
#     {'C': [1, 10, 100, 1000], 'penalty' : ['l1'], 'solver' : ['saga'] }   
# ]


#Decision Trees parameters
# parameters = [
#     { 'criterion' : ['entropy', 'gini'], 'min_samples_split' : range(10,500,50),'max_depth': range(1,20,2) }
# ]

#2nd round of tuning
# parameters = [
#     { 'criterion' : ['entropy', 'gini'], 'min_samples_split' : range(400,480,15),'max_depth': range(1,7) }
# ]

#Random forest parameters
# parameters = [
#     { 'n_estimators' : range(10, 500, 10), 'min_samples_split': [410, 160] }    #410, 160 were the two values found on DecisionTreeClassifier tunning process
# ]                                                                              #for accuracy and F1 score respectively 

parameters = [
    { 'criterion' : ['entropy', 'gini'],'max_depth': [16], 'n_estimators' : range(300, 320, 10), 'min_samples_split': range(20, 80, 20 ) }    #410, 160 were the two values found on DecisionTreeClassifier tunning process
]     

vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words=stopwords)
X_train = vectorizer.fit_transform(X_train)
print("Vectorized.")

svd = TruncatedSVD(n_components=1000,algorithm='arpack', random_state=42)
X_train = svd.fit_transform(X_train)
print("SVD performed.")


scores = [ 'f1']

for score in scores:

    print("# Tuning hyper-parameters for %s" % score)
    print()
    
    #clf = SVC()
    #clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()
    clf = GridSearchCV(clf, parameters, scoring='%s' % score, cv=5, return_train_score=True, verbose=1000, n_jobs=-1)
    
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)

dict_res = clf.cv_results_
df = pd.DataFrame()
for key in dict_res:
    df[key] = list(dict_res[key])

df.to_csv("gridsearch_RF3.csv", sep=',',index = False ,header = True)