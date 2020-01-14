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
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

#Load train data
X_train = pd.read_csv("train_gossipcop.csv", ",")
Y_train = X_train['label'].values
X_train = X_train['text'].values
print("Train set read.")
#Load dev data
# X_dev = pd.read_csv("dev_gossipcop.csv", ",")
# Y_dev = X_dev['label'].values
# X_dev = X_dev['text'].values
# print("Dev set read.")
X_test = pd.read_csv("test_gossipcop.csv", ",")
Y_test = X_test['label'].values
X_test = X_test['text'].values
print("Test set read.")

stopwords = set(ENGLISH_STOP_WORDS)

vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.25, stop_words=stopwords)  #with 0.25 on average each of 3 classifiers has its best performance
X_train = vectorizer.fit_transform(X_train)
#X_dev = vectorizer.transform(X_dev)
X_test = vectorizer.transform(X_test)
print("Vectorized.")

svd = TruncatedSVD(n_components=1000, algorithm='arpack', random_state=42)
X_train = svd.fit_transform(X_train)
#X_dev = svd.transform(X_dev)
X_test = svd.transform(X_test)

print("SVD finished.")


svm = SVC(C=10,gamma=1, kernel='rbf', probability=True)
LR = LogisticRegression(C=10, penalty='l2', solver='liblinear')
#DT = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=400)
KNN = KNeighborsClassifier(metric='minkowski', n_neighbors=5, weights='distance', p = 6)

VC = VotingClassifier(estimators=[('svm', svm), ('LR', LR), ('KNN', KNN)], voting='soft')
VC = VC.fit(X_train, Y_train)
print("Trained.")

Y_predict_train = VC._predict(X_train)
print("Train predicted.")

Y_predict_train_final = VC.predict(X_train)
mislabel = 0
_all = len(Y_predict_train)
for i in range(len(Y_predict_train)):

    if Y_predict_train_final[i] != Y_train[i]:
        if Y_train[i] in Y_predict_train[i]:    
            mislabel = mislabel + 1 

percentage = mislabel / _all
print("Percentage of mislabeled samples that one or more classifier had predicted right on train set:" + 
        str(percentage))

Y_predict_test = VC._predict(X_test)
print("test predicted.")

Y_predict_test_final = VC.predict(X_test)
mislabel = 0
_all = len(Y_predict_test)
for i in range(len(Y_predict_test)):

    
    if Y_predict_test_final[i] != Y_test[i]:
        if Y_test[i] in Y_predict_test[i]:
            mislabel = mislabel + 1 

percentage = mislabel / _all
print("Percentage of mislabeled samples that one or more classifier had predicted right on test set:" + 
        str(percentage))

print("Train accuracy: " + str(accuracy_score(Y_train, Y_predict_train_final)))
print("Train F1 score: " + str(f1_score(Y_train, Y_predict_train_final)))

print("test accuracy: " + str(accuracy_score(Y_test, Y_predict_test_final)))
print("test F1 score: " + str(f1_score(Y_test, Y_predict_test_final)))
