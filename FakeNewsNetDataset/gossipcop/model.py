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
import scikitplot as skplt

#Load train data
X_train_origin = pd.read_csv("train_gossipcop_vol2.csv", ",")
Y_train = X_train_origin['label'].values
X_train_origin = X_train_origin['text'].values
print("Train set read.")

#Load test data
X_test_origin = pd.read_csv("test_gossipcop_vol2.csv", ",")
Y_test = X_test_origin['label'].values
X_test_origin = X_test_origin['text'].values
print("Test set read.")

stopwords = set(ENGLISH_STOP_WORDS)

#SVC
print("SVM Classifier training and results:")
svm_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.29, stop_words=stopwords)
X_train = svm_vectorizer.fit_transform(X_train_origin)
X_test = svm_vectorizer.transform(X_test_origin) 

print("Vectorized.")

svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

print("SVD finished.")

svm = SVC(C=10, gamma='scale', kernel='rbf', random_state=42 ,probability=True)

svm.fit(X_train, Y_train)
print("Trained.")
Y_predict_test = svm.predict(X_test)
print("Test predicted.")

print("Test accuracy: " + str(accuracy_score(Y_test, Y_predict_test)))
print("Test F1 score: " + str(f1_score(Y_test, Y_predict_test)))

#Plot confusion matrix for the test set
# cf_matrix = confusion_matrix(Y_train, Y_predict_train, labels=[0, 1])
# htmp_train = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
# plt.title("Confusion Matrix of Train set")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()
# plt.clf()
#Plot confusion matrix for the test set
cf_matrix = confusion_matrix(Y_test, Y_predict_test, labels=[0, 1])
htmp_test = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
plt.title("SVM: Confusion Matrix of Test set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

plt.clf()

Y_probas = svm.predict_proba(X_test)
skplt.metrics.plot_precision_recall_curve(Y_test, Y_probas, title="SVM: Precision-Recall Curve" )
plt.show()
plt.clf()


# KNeighborsClassifier
print("KNN Classifier training and results:")
knn_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.56, stop_words=stopwords)
X_train = knn_vectorizer.fit_transform(X_train_origin)
X_test = knn_vectorizer.transform(X_test_origin)

print("Vectorized.")

svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

print("SVD finished.")

knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')

knn.fit(X_train, Y_train)
print("Trained.")
Y_predict_test = knn.predict(X_test)
print("Test predicted.")

print("Test accuracy: " + str(accuracy_score(Y_test, Y_predict_test)))
print("Test F1 score: " + str(f1_score(Y_test, Y_predict_test)))

#Plot confusion matrix for the test set
# cf_matrix = confusion_matrix(Y_train, Y_predict_train, labels=[0, 1])
# htmp_train = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
# plt.title("Confusion Matrix of Train set")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()
# plt.clf()
#Plot confusion matrix for the test set
cf_matrix = confusion_matrix(Y_test, Y_predict_test, labels=[0, 1])
htmp_test = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
plt.title("KNN: Confusion Matrix of Test set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

plt.clf()

Y_probas = knn.predict_proba(X_test)
skplt.metrics.plot_precision_recall_curve(Y_test, Y_probas, title="KNN: Precision-Recall Curve" )
plt.show()
plt.clf()



# LogisticRegression
print("LR Classifier training and results:")
LR_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.65, stop_words=stopwords)
X_train = LR_vectorizer.fit_transform(X_train_origin)
X_test = LR_vectorizer.transform(X_test_origin) 

print("Vectorized.")

svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

print("SVD finished.")

LR = LogisticRegression(C = 100, penalty='l1', solver='liblinear', max_iter=1000, random_state=42)

LR.fit(X_train, Y_train)
print("Trained.")
Y_predict_test = LR.predict(X_test)
print("Test predicted.")

print("Test accuracy: " + str(accuracy_score(Y_test, Y_predict_test)))
print("Test F1 score: " + str(f1_score(Y_test, Y_predict_test)))

#Plot confusion matrix for the test set
# cf_matrix = confusion_matrix(Y_train, Y_predict_train, labels=[0, 1])
# htmp_train = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
# plt.title("Confusion Matrix of Train set")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()
# plt.clf()
#Plot confusion matrix for the test set
cf_matrix = confusion_matrix(Y_test, Y_predict_test, labels=[0, 1])
htmp_test = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
plt.title("LR: Confusion Matrix of Test set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

plt.clf()


Y_probas = LR.predict_proba(X_test)
skplt.metrics.plot_precision_recall_curve(Y_test, Y_probas, title="LR: Precision-Recall Curve" )
plt.show()
plt.clf()



# DecisionTreeClassifier
print("DT Classifier training and results:")
DT_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.25, stop_words=stopwords)
X_train = DT_vectorizer.fit_transform(X_train_origin)
X_test = DT_vectorizer.transform(X_test_origin)

print("Vectorized.")

svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

print("SVD finished.")

DT = DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=420, random_state=42)

DT.fit(X_train, Y_train)
print("Trained.")
Y_predict_test = DT.predict(X_test)
print("Test predicted.")

print("Test accuracy: " + str(accuracy_score(Y_test, Y_predict_test)))
print("Test F1 score: " + str(f1_score(Y_test, Y_predict_test)))

#Plot confusion matrix for the test set
# cf_matrix = confusion_matrix(Y_train, Y_predict_train, labels=[0, 1])
# htmp_train = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
# plt.title("Confusion Matrix of Train set")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()
# plt.clf()
#Plot confusion matrix for the test set
cf_matrix = confusion_matrix(Y_test, Y_predict_test, labels=[0, 1])
htmp_test = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
plt.title("DT: Confusion Matrix of Test set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

plt.clf()

Y_probas = DT.predict_proba(X_test)
skplt.metrics.plot_precision_recall_curve(Y_test, Y_probas, title="DT: Precision-Recall Curve" )
plt.show()
plt.clf()


# RandomForestClassifier
print("RF Classifier training and results:")
RF_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.21, stop_words=stopwords)
X_train = RF_vectorizer.fit_transform(X_train_origin)
X_test = RF_vectorizer.transform(X_test_origin)

print("Vectorized.")

svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

print("SVD finished.")

RF = RandomForestClassifier(criterion='gini', max_depth=None, min_samples_split=2, n_estimators=180, random_state=42)


RF.fit(X_train, Y_train)
print("Trained.")
Y_predict_test = RF.predict(X_test)
print("Test predicted.")


print("Test accuracy: " + str(accuracy_score(Y_test, Y_predict_test)))
print("Test F1 score: " + str(f1_score(Y_test, Y_predict_test)))

#Plot confusion matrix for the training set
# cf_matrix = confusion_matrix(Y_train, Y_predict_train, labels=[0, 1])
# htmp_train = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
# plt.title("Confusion Matrix of Train set")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()
# plt.clf()
#Plot confusion matrix for the test set
cf_matrix = confusion_matrix(Y_test, Y_predict_test, labels=[0, 1])
htmp_test = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
plt.title("RF: Confusion Matrix of Test set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()
plt.clf()


Y_probas = RF.predict_proba(X_test)
skplt.metrics.plot_precision_recall_curve(Y_test, Y_probas, title="RF: Precision-Recall Curve" )
plt.show()
plt.clf()