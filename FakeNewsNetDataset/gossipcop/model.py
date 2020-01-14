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
from sklearn.metrics import precision_recall_curve
from scikitplot.metrics import plot_precision_recall_curve

#Load train data
X_train = pd.read_csv("train_gossipcop.csv", ",")
Y_train = X_train['label'].values
X_train = X_train['text'].values
print("Train set read.")
#Load dev data
X_dev = pd.read_csv("dev_gossipcop.csv", ",")
Y_dev = X_dev['label'].values
X_dev = X_dev['text'].values
print("Dev set read.")
X_test = pd.read_csv("test_gossipcop.csv", ",")
Y_test = X_test['label'].values
X_test = X_test['text'].values
print("Test set read.")

stopwords = set(ENGLISH_STOP_WORDS)

# svm_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.25, stop_words=stopwords)
# X_train = svm_vectorizer.fit_transform(X_train)
# X_dev = svm_vectorizer.transform(X_dev)
# X_test = svm_vectorizer.transform(X_test) 

# knn_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.30, stop_words=stopwords)
# X_train = knn_vectorizer.fit_transform(X_train)
# X_dev = knn_vectorizer.transform(X_dev)
# X_test = knn_vectorizer.transform(X_test) 

# LR_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.56, stop_words=stopwords)
# X_train = LR_vectorizer.fit_transform(X_train)
# X_dev = LR_vectorizer.transform(X_dev)
# X_test = LR_vectorizer.transform(X_test) 

# DT_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.75, stop_words=stopwords)
# X_train = DT_vectorizer.fit_transform(X_train)
# X_dev = DT_vectorizer.transform(X_dev)
# X_test = DT_vectorizer.transform(X_test)

RF_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.33, stop_words=stopwords)
X_train = RF_vectorizer.fit_transform(X_train)
X_dev = RF_vectorizer.transform(X_dev)
X_test = RF_vectorizer.transform(X_test) 

print("Vectorized.")

svd = TruncatedSVD(n_components=1000, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)
X_dev = svd.transform(X_dev)
X_test = svd.transform(X_test)

print("SVD finished.")

#clf = SVC(C=10, kernel='linear')
#clf = SVC(C=10,gamma=1, kernel='rbf') #2nd round of tuning

#clf = KNeighborsClassifier(metric='euclidean', n_neighbors=5, weights='distance')
#clf = KNeighborsClassifier(metric='minkowski', n_neighbors=5, weights='distance', p = 6)


#clf = LogisticRegression(C=10, penalty='l2', solver='liblinear')  

#clf = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=410)  #best params found regarding accuracy
#clf = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=160)  #best params found regarding F1_score
#clf = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=400)  #2nd tuning

#clf = RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=7, min_samples_split=160)
clf = RandomForestClassifier(n_estimators=300, criterion="gini", max_depth=16, min_samples_split=20)

clf.fit(X_train, Y_train)
print("Trained.")
Y_predict_train = clf.predict(X_train)
print("Train predicted.")
Y_predict_dev = clf.predict(X_dev)
print("Dev predicted.")
Y_predict_test = clf.predict(X_test)
print("Test predicted.")


print("Train accuracy: " + str(accuracy_score(Y_train, Y_predict_train)))
print("Train F1 score: " + str(f1_score(Y_train, Y_predict_train)))

print("Dev accuracy: " + str(accuracy_score(Y_dev, Y_predict_dev)))
print("Dev F1 score: " + str(f1_score(Y_dev, Y_predict_dev)))

print("Test accuracy: " + str(accuracy_score(Y_test, Y_predict_test)))
print("Test F1 score: " + str(f1_score(Y_test, Y_predict_test)))

#Plot confusion matrix for the training set
cf_matrix = confusion_matrix(Y_train, Y_predict_train, labels=[0, 1])
htmp_train = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
plt.title("Confusion Matrix of Train set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
plt.clf()
#Plot confusion matrix for the test set
cf_matrix = confusion_matrix(Y_test, Y_predict_test, labels=[0, 1])
htmp_test = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
plt.title("Confusion Matrix of Test set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()