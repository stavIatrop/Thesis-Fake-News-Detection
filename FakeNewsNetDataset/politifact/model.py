import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#Load train data
X_train = pd.read_csv("train_politifact.csv", ",")
Y_train = X_train['label'].values
X_train = X_train['text'].values
print("Train set read.")
#Load dev data
X_dev = pd.read_csv("dev_politifact.csv", ",")
Y_dev = X_dev['label'].values
X_dev = X_dev['text'].values
print("Dev set read.")
X_test = pd.read_csv("test_politifact.csv", ",")
Y_test = X_test['label'].values
X_test = X_test['text'].values
print("Test set read.")

stopwords = set(ENGLISH_STOP_WORDS)

svm_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.74, stop_words=stopwords)
X_train = svm_vectorizer.fit_transform(X_train)
X_dev = svm_vectorizer.transform(X_dev)
X_test = svm_vectorizer.transform(X_test) 

# knn_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.79, stop_words=stopwords)
# X_train = knn_vectorizer.fit_transform(X_train)
# X_dev = knn_vectorizer.transform(X_dev)
# X_test = knn_vectorizer.transform(X_test) 

# LR_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.56, stop_words=stopwords)
# X_train = LR_vectorizer.fit_transform(X_train)
# X_dev = LR_vectorizer.transform(X_dev)
# X_test = LR_vectorizer.transform(X_test) 

# DT_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.74, stop_words=stopwords)
# X_train = DT_vectorizer.fit_transform(X_train)
# X_dev = DT_vectorizer.transform(X_dev)
# X_test = DT_vectorizer.transform(X_test) 

# print("Vectorized.")

# vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.25, stop_words=stopwords)
# X_train = vectorizer.fit_transform(X_train)
# print(X_train.shape)
# X_dev = vectorizer.transform(X_dev)
# X_test = vectorizer.transform(X_test) 
# print("Vectorized.")


svd = TruncatedSVD(n_components=350, algorithm='arpack', random_state=42)
X_train = svd.fit_transform(X_train)
X_dev = svd.transform(X_dev)
X_test = svd.transform(X_test)

print("SVD finished.")

#clf = SVC(C = 1, kernel='linear')
#clf = SVC(C = 10, gamma = 1 ,kernel='rbf')  #2nd round of tuning
clf = SVC(C = 10, kernel='linear')  #3rd round of tuning

#clf = KNeighborsClassifier(metric='euclidean', n_neighbors = 3, weights = 'uniform')
#clf = KNeighborsClassifier(metric='minkowski', n_neighbors=1, weights='distance', p = 4) #2nd round of tuning

#clf = LogisticRegression(C=1000, penalty='l1', solver='saga')          #best params found regarding accuracy
#clf = LogisticRegression(C = 100, penalty = 'l1', solver = 'saga')          #best params found regarding F1_score


#clf = DecisionTreeClassifier(criterion= 'entropy', max_depth = 3, min_samples_split = 160)  #best params found regarding accuracy
#clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, min_samples_split = 210)  #best params found regarding F1_score

#clf = RandomForestClassifier(criterion = 'gini', max_depth = 11, min_samples_split = 10, n_estimators = 270) #best params found regarding accuracy
#clf = RandomForestClassifier(criterion = 'entropy', max_depth = 15, min_samples_split = 110, n_estimators = 160, random_state=0)
#clf = RandomForestClassifier(criterion = 'entropy', max_depth = 15, min_samples_split = 10, n_estimators = 470 ,random_state=42) #best params found regarding F1_score
#clf = RandomForestClassifier(criterion = 'entropy', max_depth = 5, min_samples_split = 120, n_estimators = 90 ,random_state=42) #best params on vol3 found regarding F1_score
#clf = RandomForestClassifier(criterion = 'entropy', max_depth = 2, min_samples_split = 160, n_estimators = 30 ,random_state=42) #best params based on validation curves

#print("criterion = 'entropy', max_depth = 15, min_samples_split = 110, n_estimators = 160")
#print("criterion = 'entropy', max_depth = 15, min_samples_split = 10, n_estimators = 470")


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
print("Train recall score: " + str(recall_score(Y_train, Y_predict_train)))

print("Dev accuracy: " + str(accuracy_score(Y_dev, Y_predict_dev)))
print("Dev F1 score: " + str(f1_score(Y_dev, Y_predict_dev)))
print("Dev recall score: " + str(recall_score(Y_dev, Y_predict_dev)))

print("Test accuracy: " + str(accuracy_score(Y_test, Y_predict_test)))
print("Test F1 score: " + str(f1_score(Y_test, Y_predict_test)))
print("Test recall score: " + str(recall_score(Y_test, Y_predict_test)))