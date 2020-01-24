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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report

X_train = pd.read_csv("train_politifact_vol2.csv", ",", usecols=['text', 'label'])
y_train = X_train['label'].values.flatten()
X_train = X_train['text'].values.flatten()
print("Read")


stopwords = set(ENGLISH_STOP_WORDS)

# svm_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.56, stop_words=stopwords)
# X_train = svm_vectorizer.fit_transform(X_train)
# print("Vectorized.")

#SVM parameters
# parameters = [  
#   {'C': [10, 100], 'kernel': ['linear']}
#  ]

# knn_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.53, stop_words=stopwords)
# X_train = knn_vectorizer.fit_transform(X_train)
# print("Vectorized.")


#KNN parameters
# parameters = [
#     {'weights' : ['uniform', 'distance'], 'metric' : ['euclidean', 'manhattan']},
#        {'weights' : ['uniform', 'distance'],'metric' : ['minkowski'], 'p' : [4]},
# ]

# parameters = [
#     {'n_neighbors': [8, 9 , 10 ] ,'weights' : ['uniform', 'distance'], 'metric' : ['euclidean', 'manhattan']},
#        {'n_neighbors': [8, 9 , 10 ] , 'weights' : ['uniform', 'distance'],'metric' : ['minkowski'], 'p' : [5, 6]}
# ]

# LR_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.33, stop_words=stopwords)
# X_train = LR_vectorizer.fit_transform(X_train)
# print("Vectorized.")


#Logistic Regression parameters
# parameters = [
#     {'C': [0.1, 1], 'penalty' : ['l2'], 'solver' : ['sag', 'saga'] },
#     {'C': [0.1, 1], 'penalty' : ['l1'], 'solver' : ['saga'] }   
# ]

# parameters = [
#     {'C': [10, 100], 'penalty' : ['l1', 'l2'], 'solver' : ['liblinear']},
#     {'C': [10, 100], 'penalty' : ['none', 'l2'], 'solver' : ['newton-cg', 'lbfgs', 'sag', 'saga'] }
# ]

# DT_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.77, stop_words=stopwords)
# X_train = DT_vectorizer.fit_transform(X_train)
# print("Vectorized.")

#Decision Trees parameters
# parameters = [
#     { 'criterion' : ['entropy', 'gini'], 'min_samples_split' : range(10,201, 20),'max_depth': range(1,4) }
# ]

# parameters = [
#     { 'criterion' : ['entropy', 'gini'], 'min_samples_split' : range(300, 401, 200),'max_depth': [1, 2, 3, 4 ,5, 6] }
# ]

RF_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.32, stop_words=stopwords)
X_train = RF_vectorizer.fit_transform(X_train)
print("Vectorized.")


#Random forest parameters
# parameters = [
#     { 'n_estimators' : range(10, 101, 10), 'criterion' : ['entropy', 'gini'], 'min_samples_split' : range(150,231,10),'max_depth': [2] } 
# ]                                                                              

parameters = [
    { 'n_estimators' : range(10, 101, 10), 'criterion' : ['entropy', 'gini'], 'min_samples_split' : range(10,101,10),'max_depth': [1, 3 ,6 ,10] } 
]                                                                              

svd = TruncatedSVD(n_components=50,algorithm='arpack', random_state=42)
X_train = svd.fit_transform(X_train)
print("SVD performed.")

scores = ['f1']

for score in scores:

    print("# Tuning hyper-parameters for %s" % score)
    print()

    #clf = SVC(random_state=42)
    #clf = KNeighborsClassifier()
    #clf = LogisticRegression(random_state=42)
    #clf = DecisionTreeClassifier(random_state=42)
    clf = RandomForestClassifier(random_state=42)

    clf = GridSearchCV(clf, parameters, scoring='%s' % score, cv=5, return_train_score=True,n_jobs=-1, verbose=1000 )
    
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    #print("Grid scores on development set:")
    # print()
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # print()

    # print("Detailed classification report:")
    # print()
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # print()
    # y_true, y_pred = y_dev, clf.predict(X_dev)
    # print(classification_report(y_true, y_pred))
    # print()

dict_res = clf.cv_results_
df = pd.DataFrame()
for key in dict_res:
    df[key] = list(dict_res[key])

df.to_csv("gridsearch_RF4.csv", sep=',',index = False ,header = True)