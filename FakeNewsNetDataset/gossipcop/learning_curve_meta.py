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
from sklearn.model_selection import validation_curve


#Load train data
X_train_origin = pd.read_csv("train_gossipcop_vol2.csv", ",")
Y_train = X_train_origin['label'].values
X_train_origin = X_train_origin['text'].values
print("Train set read.")


stopwords = set(ENGLISH_STOP_WORDS)

#SVC
print("SVM Classifier training and results:")
svm_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.29, stop_words=stopwords)
X_train = svm_vectorizer.fit_transform(X_train_origin)

print("Vectorized.")

svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)

print("SVD finished.")

svm = SVC(C=10, gamma='scale', kernel='rbf', random_state=42 ,probability=True)

svm.fit(X_train, Y_train)
print("Trained.")
Y_probas_train_svm = svm.predict_proba(X_train)
print("Probabilities predicted.")

# KNeighborsClassifier
print("KNN Classifier training and results:")
knn_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.56, stop_words=stopwords)
X_train = knn_vectorizer.fit_transform(X_train_origin)

print("Vectorized.")

svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)

print("SVD finished.")

knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')

knn.fit(X_train, Y_train)
print("Trained.")
Y_probas_train_knn = knn.predict_proba(X_train)

print("Probabilities predicted.")

# LogisticRegression
print("LR Classifier training and results:")
LR_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.65, stop_words=stopwords)
X_train = LR_vectorizer.fit_transform(X_train_origin)

print("Vectorized.")

svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)

print("SVD finished.")

LR = LogisticRegression(C = 100, penalty='l1', solver='liblinear', max_iter=1000, random_state=42)

LR.fit(X_train, Y_train)
print("Trained.")
Y_probas_train_LR = LR.predict_proba(X_train)

print("Probabilities predicted.")

# DecisionTreeClassifier
print("DT Classifier training and results:")
DT_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.25, stop_words=stopwords)
X_train = DT_vectorizer.fit_transform(X_train_origin)

print("Vectorized.")

svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)

print("SVD finished.")

DT = DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=420, random_state=42)

DT.fit(X_train, Y_train)
print("Trained.")
Y_probas_train_DT = DT.predict_proba(X_train)

print("Probabilities predicted.")


# RandomForestClassifier
print("RF Classifier training and results:")
RF_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.21, stop_words=stopwords)
X_train = RF_vectorizer.fit_transform(X_train_origin)

print("Vectorized.")

svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)

print("SVD finished.")

RF = RandomForestClassifier(criterion='gini', max_depth=None, min_samples_split=2, n_estimators=180, random_state=42)


RF.fit(X_train, Y_train)
print("Trained.")
Y_probas_train_RF = RF.predict_proba(X_train)

print("Probabilities predicted.")

#Ensemble Classifier
Y_class1_train_svm = Y_probas_train_svm[np.newaxis, :, 1].T     #each one with shape (m, 1), m=number of training instances
Y_class1_train_knn = Y_probas_train_knn[np.newaxis, :, 1].T
Y_class1_train_LR = Y_probas_train_LR[np.newaxis, :, 1].T
Y_class1_train_DT = Y_probas_train_DT[np.newaxis, :, 1].T
Y_class1_train_RF = Y_probas_train_RF[np.newaxis, :, 1].T

X_meta_train = np.concatenate((Y_class1_train_svm, Y_class1_train_knn, Y_class1_train_LR,  Y_class1_train_DT, Y_class1_train_RF), axis=1)  #concatenate horizontally, final shape (m, 5)
Y_meta_train = Y_train


x = [0.1, 1, 10 , 100, 1000, 1100, 1300, 1500]
train_scores, valid_scores = validation_curve(LogisticRegression(random_state=42), X_train, Y_meta_train, "C", x ,   cv=3 , verbose=1000, n_jobs=-1, scoring='accuracy')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("C values")
plt.ylabel("accuracy Score")
plt.axis([0, max(x) + 0.001, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve LR-C")
plt.legend()

plt.show()

plt.clf()

x = [1, 2, 3, 4, 5]
train_scores, valid_scores = validation_curve(LogisticRegression(random_state=42), X_meta_train, Y_meta_train, "solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] ,   cv=3 , verbose=1000, n_jobs=-1, scoring='accuracy')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("Solver")
plt.ylabel("accuracy Score")
plt.axis([0, max(x) + 0.001, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve LR-Solver")
plt.legend()

plt.show()

plt.clf()




x = [0.1, 1, 10 , 100, 1000, 1100, 1300, 1500]
train_scores, valid_scores = validation_curve(LogisticRegression(random_state=42), X_meta_train, Y_meta_train, "C", x ,   cv=3 , verbose=1000, n_jobs=-1, scoring='f1')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("C values")
plt.ylabel("F1 Score")
plt.axis([0, max(x) + 0.001, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve LR-C")
plt.legend()

plt.show()

plt.clf()

x = [1, 2, 3, 4, 5]
train_scores, valid_scores = validation_curve(LogisticRegression(random_state=42), X_meta_train, Y_meta_train, "solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] ,   cv=3 , verbose=1000, n_jobs=-1, scoring='f1')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("Solver")
plt.ylabel("F1 Score")
plt.axis([0, max(x) + 0.001, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve LR-Solver")
plt.legend()

plt.show()

plt.clf()