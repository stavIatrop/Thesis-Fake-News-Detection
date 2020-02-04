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
X_train_origin = pd.read_csv("train_politifact_vol2.csv", ",")
Y_train = X_train_origin['label'].values
X_train_origin = X_train_origin['text'].values
print("Train set read.")

#Load test data
X_test_origin = pd.read_csv("test_politifact_vol2.csv", ",")
Y_test = X_test_origin['label'].values
X_test_origin = X_test_origin['text'].values
print("Test set read.")

stopwords = set(ENGLISH_STOP_WORDS)

#SVC
print("SVM Classifier training and results:")
svm_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.56, stop_words=stopwords)
X_train = svm_vectorizer.fit_transform(X_train_origin)
X_test = svm_vectorizer.transform(X_test_origin) 

print("Vectorized.")

svd = TruncatedSVD(n_components=50, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

print("SVD finished.")

svm = SVC(C=10, kernel='linear', random_state=42, probability=True)

svm.fit(X_train, Y_train)
print("Trained.")
Y_probas_train_svm = svm.predict_proba(X_train)
Y_probas_test_svm = svm.predict_proba(X_test)
print("Probabilities predicted.")

# KNeighborsClassifier
print("KNN Classifier training and results:")
knn_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.53, stop_words=stopwords)
X_train = knn_vectorizer.fit_transform(X_train_origin)
X_test = knn_vectorizer.transform(X_test_origin)

print("Vectorized.")

svd = TruncatedSVD(n_components=50, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

print("SVD finished.")

knn = KNeighborsClassifier(n_neighbors=10, weights='distance', metric='manhattan')

knn.fit(X_train, Y_train)
print("Trained.")
Y_probas_train_knn = knn.predict_proba(X_train)
Y_probas_test_knn = knn.predict_proba(X_test)

print("Probabilities predicted.")

# LogisticRegression
print("LR Classifier training and results:")
LR_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.33, stop_words=stopwords)
X_train = LR_vectorizer.fit_transform(X_train_origin)
X_test = LR_vectorizer.transform(X_test_origin) 

print("Vectorized.")

svd = TruncatedSVD(n_components=50, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

print("SVD finished.")

LR = LogisticRegression(C = 100, penalty='l2', solver='saga', max_iter=1000, random_state=42)

LR.fit(X_train, Y_train)
print("Trained.")
Y_probas_train_LR = LR.predict_proba(X_train)
Y_probas_test_LR = LR.predict_proba(X_test)

print("Probabilities predicted.")

# DecisionTreeClassifier
print("DT Classifier training and results:")
DT_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.77, stop_words=stopwords)
X_train = DT_vectorizer.fit_transform(X_train_origin)
X_test = DT_vectorizer.transform(X_test_origin)

print("Vectorized.")

svd = TruncatedSVD(n_components=50, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

print("SVD finished.")

DT = DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_split=300, random_state=42)

DT.fit(X_train, Y_train)
print("Trained.")
Y_probas_train_DT = DT.predict_proba(X_train)
Y_probas_test_DT = DT.predict_proba(X_test)

print("Probabilities predicted.")


# RandomForestClassifier
print("RF Classifier training and results:")
RF_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.32, stop_words=stopwords)
X_train = RF_vectorizer.fit_transform(X_train_origin)
X_test = RF_vectorizer.transform(X_test_origin)

print("Vectorized.")

svd = TruncatedSVD(n_components=50, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

print("SVD finished.")

RF = RandomForestClassifier(criterion='gini', max_depth=10, min_samples_split=10, n_estimators=100, random_state=42)


RF.fit(X_train, Y_train)
print("Trained.")
Y_probas_train_RF = RF.predict_proba(X_train)
Y_probas_test_RF = RF.predict_proba(X_test)

print("Probabilities predicted.")

#Ensemble Classifier
Y_class1_train_svm = Y_probas_train_svm[np.newaxis, :, 1].T     #each one with shape (m, 1), m=number of training instances
Y_class1_train_knn = Y_probas_train_knn[np.newaxis, :, 1].T
Y_class1_train_LR = Y_probas_train_LR[np.newaxis, :, 1].T
Y_class1_train_DT = Y_probas_train_DT[np.newaxis, :, 1].T
Y_class1_train_RF = Y_probas_train_RF[np.newaxis, :, 1].T

Y_class1_test_svm = Y_probas_test_svm[np.newaxis, :, 1].T     #each one with shape (n, 1), n=number of test instances
Y_class1_test_knn = Y_probas_test_knn[np.newaxis, :, 1].T
Y_class1_test_LR = Y_probas_test_LR[np.newaxis, :, 1].T
Y_class1_test_DT = Y_probas_test_DT[np.newaxis, :, 1].T
Y_class1_test_RF = Y_probas_test_RF[np.newaxis, :, 1].T

X_meta_train = np.concatenate((Y_class1_train_svm, Y_class1_train_knn, Y_class1_train_LR,  Y_class1_train_DT, Y_class1_train_RF), axis=1)  #concatenate horizontally, final shape (m, 5)
Y_meta_train = Y_train

X_meta_test = np.concatenate((Y_class1_test_svm, Y_class1_test_knn, Y_class1_test_LR, Y_class1_test_DT, Y_class1_test_RF), axis=1)  #concatenate horizontally, final shape (n, 5)
Y_meta_test = Y_test

meta_clf = LogisticRegression(random_state=42)
meta_clf.fit(X_meta_train, Y_meta_train)
print("Meta-classifier trained.")

Y_predict_meta = meta_clf.predict(X_meta_test)
print(meta_clf.coef_)
print("Test accuracy: " + str(accuracy_score(Y_meta_test, Y_predict_meta)))
print("Test F1 score: " + str(f1_score(Y_meta_test, Y_predict_meta)))


#Plot confusion matrix for the test set
cf_matrix = confusion_matrix(Y_meta_test, Y_predict_meta, labels=[0, 1])
htmp_test = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
plt.title("Meta_Classifier: Confusion Matrix of Test set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()
plt.clf()


Y_meta_probas = meta_clf.predict_proba(X_meta_test)
skplt.metrics.plot_precision_recall_curve(Y_meta_test, Y_meta_probas, title="Meta_Classifier: Precision-Recall Curve" )
plt.show()
plt.clf()