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

X_train = pd.read_csv("train_politifact.csv", ",", usecols=['text', 'label'])
y_train = X_train['label'].values.flatten()
X_train = X_train['text'].values.flatten()
print("Read")
X_dev = pd.read_csv("dev_politifact.csv", ",", usecols=['text', 'label'])
y_dev = X_dev['label'].values.flatten()
X_dev = X_dev['text'].values.flatten()
print("Read")

stopwords = set(ENGLISH_STOP_WORDS)

#SVM parameters
parameters = [  
  {'C': [10, 100], 'kernel': ['linear']}
 ]

#KNN parameters
# parameters = {
#     'n_neighbors' : [3, 5, 11],
#     'weights' : ['uniform', 'distance'],
#     'metric' : ['euclidean', 'manhattan']
# }

#Logistic Regression parameters
# parameters = [
#     {'C': [0.1, 1], 'penalty' : ['l2'], 'solver' : ['sag', 'saga'] },
#     {'C': [0.1, 1], 'penalty' : ['l1'], 'solver' : ['saga'] }   
# ]

#Decision Trees parameters
# parameters = [
#     { 'criterion' : ['entropy', 'gini'], 'min_samples_split' : range(120,200,25),'max_depth': range(1,3) }
# ]

#Random forest parameters
# parameters = [
#     { 'n_estimators' : range(10, 100, 10), 'criterion' : ['entropy', 'gini'], 'min_samples_split' : range(100,220,20),'max_depth': range(1,10,2) }    #410, 160 were the two values found on DecisionTreeClassifier tunning process
# ]                                                                              #for accuracy and F1 score respectively 


vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words=stopwords)
X_train = vectorizer.fit_transform(X_train)
X_dev = vectorizer.transform(X_dev)
print("Vectorized.")

svd = TruncatedSVD(n_components=350,algorithm='arpack', random_state=42)
X_train = svd.fit_transform(X_train)
X_dev = svd.transform(X_dev)
print("SVD performed.")

scores = ['f1']

for score in scores:

    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = SVC()
    #clf = KNeighborsClassifier()
    #clf = LogisticRegression()
    #clf = DecisionTreeClassifier()
    #clf = RandomForestClassifier(random_state=42)

    clf = GridSearchCV(clf, parameters, scoring='%s' % score, cv=5, return_train_score=True, verbose=100 )
    
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
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

df.to_csv("gridsearch_svm2.csv", sep=',',index = False ,header = True)