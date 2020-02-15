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
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from numpy import interp

#Load train data
X_origin = pd.read_csv("train_isot.csv", ",")
Y = X_origin['label'].values
X_origin = X_origin['text'].values
print("Train set read.")

stopwords = set(ENGLISH_STOP_WORDS)

svm_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.73, stop_words=stopwords)
X = svm_vectorizer.fit_transform(X_origin)

print("Vectorized.")

svd = TruncatedSVD(n_components=200, algorithm='arpack', random_state=42)
print("SVD prepared.")
X = svd.fit_transform(X)


print("SVD finished.")
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)

# fig, ax = plt.subplots()
# score_f = 0
# score_a = 0

# kf = KFold(n_splits=5,random_state=42, shuffle=True)
# for i, (train, test) in enumerate(kf.split(X)):
#     X_train = X[train]
#     X_test = X[test]
#     Y_train = Y[train]
#     Y_test = Y[test]
 
#     #clf = SVC(random_state=42) 
#     clf = SVC(C=10, gamma=10, kernel='rbf', random_state=42, probability=True) 
    
#     clf.fit(X_train,Y_train)
#     Y_predicted = clf.predict(X_test)
    
#     score_f += f1_score(Y_test,Y_predicted)
#     score_a += accuracy_score(Y_test,Y_predicted)

#     viz = plot_roc_curve(clf, X_test, Y_test,
#                          name='ROC fold {}'.format(i),
#                          alpha=0.3, lw=1, ax=ax)
#     interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
#     interp_tpr[0] = 0.0
#     tprs.append(interp_tpr)
#     aucs.append(viz.roc_auc)

# ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#         label='Chance', alpha=.8)

# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# ax.plot(mean_fpr, mean_tpr, color='b',
#         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#         lw=2, alpha=.8)

# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                 label=r'$\pm$ 1 std. dev.')

# ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
#        title="Receiver operating characteristic example")
# ax.legend(loc="lower right")
# plt.show()


# score_f /= 5
# score_a /= 5

# print("SVM Accuracy: " + str(score_a))
# print("SVM F1 score: " + str(score_f))



# knn_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.56, stop_words=stopwords)
# X = knn_vectorizer.fit_transform(X_origin)


# print("Vectorized.")

# svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
# print("SVD prepared.")
# X = svd.fit_transform(X)

# print("SVD finished.")

# score_f = 0
# score_a = 0

# kf = KFold(n_splits=5,random_state=42, shuffle=True)
# for train, test in kf.split(X):
#     X_train = X[train]
#     X_test = X[test]
#     Y_train = Y[train]
#     Y_test = Y[test]
 
#     #clf = KNeighborsClassifier() 
#     clf = KNeighborsClassifier(n_neighbors = 4, weights='distance', metric='minkowski' ,p = 6)
#     clf.fit(X_train,Y_train)
#     Y_predicted = clf.predict(X_test)
    
#     score_f += f1_score(Y_test,Y_predicted)
#     score_a += accuracy_score(Y_test,Y_predicted)


# score_f /= 5
# score_a /= 5

# print("KNN Accuracy: " + str(score_a))
# print("KNN F1 score: " + str(score_f))



# LR_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.65, stop_words=stopwords)
# X = LR_vectorizer.fit_transform(X_origin)


# print("Vectorized.")

# svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
# print("SVD prepared.")
# X = svd.fit_transform(X)

# print("SVD finished.")

# score_f = 0
# score_a = 0

# kf = KFold(n_splits=5,random_state=42, shuffle=True)
# for train, test in kf.split(X):
#     X_train = X[train]
#     X_test = X[test]
#     Y_train = Y[train]
#     Y_test = Y[test]
 
#     #clf = LogisticRegression(random_state=42) 
#     clf = LogisticRegression(C = 10, penalty='l1', solver='saga', max_iter=1000, random_state=42)
#     clf.fit(X_train,Y_train)
#     Y_predicted = clf.predict(X_test)
    
#     score_f += f1_score(Y_test,Y_predicted)
#     score_a += accuracy_score(Y_test,Y_predicted)


# score_f /= 5
# score_a /= 5

# print("LR Accuracy: " + str(score_a))
# print("LR F1 score: " + str(score_f))


# DT_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.25, stop_words=stopwords)
# X = DT_vectorizer.fit_transform(X_origin)
# print("Vectorized.")

# svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
# print("SVD prepared.")
# X = svd.fit_transform(X)


# print("SVD finished.")

# score_f = 0
# score_a = 0

# kf = KFold(n_splits=5,random_state=42, shuffle=True)
# for train, test in kf.split(X):
#     X_train = X[train]
#     X_test = X[test]
#     Y_train = Y[train]
#     Y_test = Y[test]
 
#     #clf = DecisionTreeClassifier(random_state=42) 
#     clf = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=10, random_state=42)
#     clf.fit(X_train,Y_train)
#     Y_predicted = clf.predict(X_test)
    
#     score_f += f1_score(Y_test,Y_predicted)
#     score_a += accuracy_score(Y_test,Y_predicted)


# score_f /= 5
# score_a /= 5

# print("DT Accuracy: " + str(score_a))
# print("DT F1 score: " + str(score_f))



# RF_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.21, stop_words=stopwords)
# X = RF_vectorizer.fit_transform(X_origin)
# print("Vectorized.")

# svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
# print("SVD prepared.")
# X = svd.fit_transform(X)


# print("SVD finished.")

score_f = 0
score_a = 0

kf = KFold(n_splits=5,random_state=42, shuffle=True)
for train, test in kf.split(X):
    X_train = X[train]
    X_test = X[test]
    Y_train = Y[train]
    Y_test = Y[test]
 
    #clf = RandomForestClassifier(random_state=42) 
    clf = RandomForestClassifier(criterion='entropy', max_depth=21, min_samples_split=6, n_estimators=350, random_state=42) 
    
    clf.fit(X_train,Y_train)
    Y_predicted = clf.predict(X_test)
    
    score_f += f1_score(Y_test,Y_predicted)
    score_a += accuracy_score(Y_test,Y_predicted)


score_f /= 5
score_a /= 5

print("RF Accuracy: " + str(score_a))
print("RF F1 score: " + str(score_f))