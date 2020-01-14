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
X_train = pd.read_csv("train_gossipcop.csv", ",", nrows=1000)
Y_train = X_train['label'].values
X_train = X_train['text'].values
print("Train set read.")
#Load dev data
X_dev = pd.read_csv("dev_gossipcop.csv", ",", nrows=1000)
Y_dev = X_dev['label'].values
X_dev = X_dev['text'].values
print("Dev set read.")
# X_test = pd.read_csv("test_gossipcop.csv", ",")
# Y_test = X_test['label'].values
# X_test = X_test['text'].values
# print("Test set read.")

stopwords = set(ENGLISH_STOP_WORDS)

vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.50, stop_words=stopwords)  #with 0.5 on average each of 3 classifiers has its best performance
X_train = vectorizer.fit_transform(X_train)
X_dev = vectorizer.transform(X_dev)
print("Vectorized.")

svd = TruncatedSVD(n_components=10, algorithm='arpack', random_state=42)
X_train = svd.fit_transform(X_train)
X_dev = svd.transform(X_dev)
# X_test = svd.transform(X_test)

print("SVD finished.")


svm = SVC(C=10, kernel='linear', probability=True)
LR = LogisticRegression(C=10, penalty='l1', solver='saga')
DT = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=410)

svm_score = svm.fit(X_train, Y_train).score(X_train, Y_train)
LR_score = LR.fit(X_train, Y_train).score(X_train, Y_train)
DT_score = DT.fit(X_train, Y_train).score(X_train, Y_train)

svm_predict = svm.predict(X_train)
LR_predict = LR.predict(X_train)
DT_predict = DT.predict(X_train)

svm_w = 0
LR_w = 0
DT_w = 0
mv = 0

svm_predict_dev = svm.predict(X_dev)
LR_predict_dev = LR.predict(X_dev)
DT_predict_dev = DT.predict(X_dev)
Y_predict_dev = list()
for i in range(len(svm_predict_dev)):

    if DT_predict[i] == svm_predict[i] or  DT_predict[i] == LR_predict[i]:
        Y_predict_dev.append(DT_predict[i])
    else:
        Y_predict_dev.append(LR_predict[i])        

print("Dev accuracy: " + str(accuracy_score(Y_dev, Y_predict_dev)))
print("Dev F1 score: " + str(f1_score(Y_dev, Y_predict_dev)))