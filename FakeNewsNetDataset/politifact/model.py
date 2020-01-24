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
X_train = pd.read_csv("train_politifact_vol2.csv", ",")
Y_train = X_train['label'].values
X_train = X_train['text'].values
print("Train set read.")
#Load dev data
# X_dev = pd.read_csv("dev_politifact.csv", ",")
# Y_dev = X_dev['label'].values
# X_dev = X_dev['text'].values
# print("Dev set read.")
X_test = pd.read_csv("test_politifact_vol2.csv", ",")
Y_test = X_test['label'].values
X_test = X_test['text'].values
print("Test set read.")

stopwords = set(ENGLISH_STOP_WORDS)

# svm_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.56, stop_words=stopwords)
# X_train = svm_vectorizer.fit_transform(X_train)

# X_test = svm_vectorizer.transform(X_test) 

# knn_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.53, stop_words=stopwords)
# X_train = knn_vectorizer.fit_transform(X_train)
# #X_dev = knn_vectorizer.transform(X_dev)
# X_test = knn_vectorizer.transform(X_test) 

# LR_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.33, stop_words=stopwords)
# X_train = LR_vectorizer.fit_transform(X_train)
# #X_dev = LR_vectorizer.transform(X_dev)
# X_test = LR_vectorizer.transform(X_test) 

# DT_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.77, stop_words=stopwords)
# X_train = DT_vectorizer.fit_transform(X_train)
# #X_dev = DT_vectorizer.transform(X_dev)
# X_test = DT_vectorizer.transform(X_test) 

RF_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.32, stop_words=stopwords)
X_train = RF_vectorizer.fit_transform(X_train)
#X_dev = RF_vectorizer.transform(X_dev)
X_test = RF_vectorizer.transform(X_test) 

# print("Vectorized.")

# vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.25, stop_words=stopwords)
# X_train = vectorizer.fit_transform(X_train)
# #X_dev = vectorizer.transform(X_dev)
# X_test = vectorizer.transform(X_test) 
# print("Vectorized.")

print(min(X_train.shape))
svd = TruncatedSVD(n_components=50, algorithm='arpack', random_state=42)
X_train = svd.fit_transform(X_train)
#X_dev = svd.transform(X_dev)
X_test = svd.transform(X_test)

print("SVD finished.")

#clf = SVC(C = 1, kernel='linear')
#clf = SVC(C = 10, gamma = 1 ,kernel='rbf')  #2nd round of tuning
#clf = SVC(C = 10, kernel='linear', random_state=42)  #3rd round of tuning

#clf = KNeighborsClassifier(metric='euclidean', n_neighbors = 3, weights = 'uniform')
#clf = KNeighborsClassifier(metric='minkowski', n_neighbors=1, weights='uniform', p = 4) #2nd round of tuning
#clf = KNeighborsClassifier(n_neighbors=10, weights='distance', metric='manhattan')

#clf = LogisticRegression(C=1000, penalty='l1', solver='saga')          #best params found regarding accuracy
#clf = LogisticRegression(C = 100, penalty = 'l1', solver = 'saga')          #best params found regarding F1_score
#clf = LogisticRegression(C = 100, penalty = 'l2', solver = 'liblinear', random_state=1)          #3rd tuning
#clf = LogisticRegression(C = 10, penalty = 'none', solver = 'sag', random_state=42)          #4th tuning
#clf = LogisticRegression(C = 100, penalty='l2', solver='saga', max_iter=1000, random_state=42)

#clf = DecisionTreeClassifier(criterion= 'entropy', max_depth = 3, min_samples_split = 160)  #best params found regarding accuracy
#clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, min_samples_split = 210)  #best params found regarding F1_score
#clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, min_samples_split = 70, random_state=42)  #3rd tuning
#clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, min_samples_split = 42, random_state=42)  #4th tuning
#clf = DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_split=300, random_state=42)

#clf = RandomForestClassifier(criterion = 'gini', max_depth = 11, min_samples_split = 10, n_estimators = 270,random_state=1) #best params found regarding accuracy
#clf = RandomForestClassifier(criterion = 'entropy', max_depth = 15, min_samples_split = 110, n_estimators = 160, random_state=1)
#clf = RandomForestClassifier(criterion = 'entropy', max_depth = 15, min_samples_split = 10, n_estimators = 470 ,random_state=1) #best params found regarding F1_score
#clf = RandomForestClassifier(criterion = 'entropy', max_depth = 5, min_samples_split = 120, n_estimators = 90 ,random_state=1) #best params on vol3 found regarding F1_score
#clf = RandomForestClassifier(criterion = 'entropy', max_depth = 2, min_samples_split = 170, n_estimators = 30 ,random_state=42) #BEST params based on validation curves
#clf = RandomForestClassifier(criterion = 'entropy', max_depth = 7, min_samples_split = 62, n_estimators = 360,random_state=1) #3rd tun
#clf = RandomForestClassifier(criterion = 'entropy', max_depth = 6, min_samples_split = 15, n_estimators = 110,random_state=42) #3rd tun
#clf = RandomForestClassifier(criterion = 'entropy', max_depth = None, min_samples_split = 52, n_estimators = 60,random_state=42) #4th tun
#clf = RandomForestClassifier(criterion = 'entropy', max_depth = 2, min_samples_split = 150, n_estimators = 50,random_state=42) #4th tun
clf = RandomForestClassifier(criterion='gini', max_depth=10, min_samples_split=10, n_estimators=100, random_state=42) 

#clf = SVC(random_state=42)
clf.fit(X_train, Y_train)
print("Trained.")
Y_predict_train = clf.predict(X_train)
print("Train predicted.")
# Y_predict_dev = clf.predict(X_dev)
# print("Dev predicted.")
Y_predict_test = clf.predict(X_test)
print("Test predicted.")


print("Train accuracy: " + str(accuracy_score(Y_train, Y_predict_train)))
print("Train F1 score: " + str(f1_score(Y_train, Y_predict_train)))
print("Train recall score: " + str(recall_score(Y_train, Y_predict_train)))

# print("Dev accuracy: " + str(accuracy_score(Y_dev, Y_predict_dev)))
# print("Dev F1 score: " + str(f1_score(Y_dev, Y_predict_dev)))
# print("Dev recall score: " + str(recall_score(Y_dev, Y_predict_dev)))

print("Test accuracy: " + str(accuracy_score(Y_test, Y_predict_test)))
print("Test F1 score: " + str(f1_score(Y_test, Y_predict_test)))
print("Test recall score: " + str(recall_score(Y_test, Y_predict_test)))