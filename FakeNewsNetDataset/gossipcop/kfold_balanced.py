import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold

#Load train data
X_origin = pd.read_csv("train_gossipcop_balanced.csv", ",")
Y = X_origin['label'].values
X_origin = X_origin['text'].values
print("Train set read.")
#Load dev data
# X_dev = pd.read_csv("dev_gossipcop_vol2.csv", ",")
# Y_dev = X_dev['label'].values
# X_dev = X_dev['text'].values
# print("Dev set read.")


stopwords = set(ENGLISH_STOP_WORDS)

svm_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.37, stop_words=stopwords)
X = svm_vectorizer.fit_transform(X_origin)

print("Vectorized.")

svd = TruncatedSVD(n_components=200, algorithm='arpack', random_state=42)
print("SVD prepared.")
X = svd.fit_transform(X)


print("SVD finished.")

score_f = 0
score_a = 0

kf = KFold(n_splits=5,random_state=42, shuffle=True)
for train, test in kf.split(X):
    X_train = X[train]
    X_test = X[test]
    Y_train = Y[train]
    Y_test = Y[test]
 
    #clf = SVC(random_state=42) 
    clf = SVC(C=1, gamma=10, kernel='rbf', random_state=42) 
    
    clf.fit(X_train,Y_train)
    Y_predicted = clf.predict(X_test)
    
    score_f += f1_score(Y_test,Y_predicted)
    score_a += accuracy_score(Y_test,Y_predicted)


score_f /= 5
score_a /= 5

print("SVM Accuracy: " + str(score_a))
print("SVM F1 score: " + str(score_f))



knn_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.72, stop_words=stopwords)
X = knn_vectorizer.fit_transform(X_origin)


print("Vectorized.")

svd = TruncatedSVD(n_components=200, algorithm='arpack', random_state=42)
print("SVD prepared.")
X = svd.fit_transform(X)

print("SVD finished.")

score_f = 0
score_a = 0

kf = KFold(n_splits=5,random_state=42, shuffle=True)
for train, test in kf.split(X):
    X_train = X[train]
    X_test = X[test]
    Y_train = Y[train]
    Y_test = Y[test]
 
    #clf = KNeighborsClassifier() 
    clf = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p = 4 )
    clf.fit(X_train,Y_train)
    Y_predicted = clf.predict(X_test)
    
    score_f += f1_score(Y_test,Y_predicted)
    score_a += accuracy_score(Y_test,Y_predicted)


score_f /= 5
score_a /= 5

print("KNN Accuracy: " + str(score_a))
print("KNN F1 score: " + str(score_f))



LR_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.22, stop_words=stopwords)
X = LR_vectorizer.fit_transform(X_origin)


print("Vectorized.")

svd = TruncatedSVD(n_components=200, algorithm='arpack', random_state=42)
print("SVD prepared.")
X = svd.fit_transform(X)

print("SVD finished.")

score_f = 0
score_a = 0

kf = KFold(n_splits=5,random_state=42, shuffle=True)
for train, test in kf.split(X):
    X_train = X[train]
    X_test = X[test]
    Y_train = Y[train]
    Y_test = Y[test]
 
    clf = LogisticRegression(random_state=42) 
    #clf = LogisticRegression(C = 100, penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
    clf.fit(X_train,Y_train)
    Y_predicted = clf.predict(X_test)
    
    score_f += f1_score(Y_test,Y_predicted)
    score_a += accuracy_score(Y_test,Y_predicted)


score_f /= 5
score_a /= 5

print("LR Accuracy: " + str(score_a))
print("LR F1 score: " + str(score_f))


DT_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.8, stop_words=stopwords)
X = DT_vectorizer.fit_transform(X_origin)
print("Vectorized.")

svd = TruncatedSVD(n_components=200, algorithm='arpack', random_state=42)
print("SVD prepared.")
X = svd.fit_transform(X)


# print("SVD finished.")

score_f = 0
score_a = 0

kf = KFold(n_splits=5,random_state=42, shuffle=True)
for train, test in kf.split(X):
    X_train = X[train]
    X_test = X[test]
    Y_train = Y[train]
    Y_test = Y[test]
 
    clf = DecisionTreeClassifier(random_state=42) 
    #clf = DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=420, random_state=42)
    clf.fit(X_train,Y_train)
    Y_predicted = clf.predict(X_test)
    
    score_f += f1_score(Y_test,Y_predicted)
    score_a += accuracy_score(Y_test,Y_predicted)


score_f /= 5
score_a /= 5

print("DT Accuracy: " + str(score_a))
print("DT F1 score: " + str(score_f))



RF_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.4, stop_words=stopwords)
X = RF_vectorizer.fit_transform(X_origin)
print("Vectorized.")

svd = TruncatedSVD(n_components=200, algorithm='arpack', random_state=42)
print("SVD prepared.")
X = svd.fit_transform(X)


print("SVD finished.")

score_f = 0
score_a = 0

kf = KFold(n_splits=5,random_state=42, shuffle=True)
for train, test in kf.split(X):
    X_train = X[train]
    X_test = X[test]
    Y_train = Y[train]
    Y_test = Y[test]
 
    clf = RandomForestClassifier(random_state=42) 
    #clf = RandomForestClassifier(criterion='gini', max_depth=None, min_samples_split=2, n_estimators=180, random_state=42) 
    
    clf.fit(X_train,Y_train)
    Y_predicted = clf.predict(X_test)
    
    score_f += f1_score(Y_test,Y_predicted)
    score_a += accuracy_score(Y_test,Y_predicted)


score_f /= 5
score_a /= 5

print("RF Accuracy: " + str(score_a))
print("RF F1 score: " + str(score_f))
