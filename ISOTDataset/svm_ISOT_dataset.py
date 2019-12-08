from sklearn import preprocessing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.parsing.porter import PorterStemmer
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import load_dataset

X_train, Y_train = load_dataset.load_dataset("train_isot.csv", ",")
X_dev, Y_dev = load_dataset.load_dataset("dev_isot.csv", ",")
#Stopwords removal

stopwords = set(ENGLISH_STOP_WORDS)
stopwords.add("said")
stopwords.add("say")
stopwords.add("says")

vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.49,  stop_words=stopwords)

X_train = vectorizer.fit_transform(X_train) 
X_dev = vectorizer.transform(X_dev) 

svd = TruncatedSVD(n_components=1000,algorithm='randomized',random_state = 42)
X_train = svd.fit_transform(X_train)
X_dev = svd.transform(X_dev)

clf = SVC(C=10.0, kernel='linear')

clf.fit(X_train, Y_train)

print("Trained.")
Y_predict = clf.predict(X_dev)

print("Precision: " + str(precision_score(Y_dev, Y_predict, average='macro')))
print("Recall: " + str(recall_score(Y_dev, Y_predict, average='macro')))
print("Accuracy: " + str(accuracy_score(Y_dev, Y_predict)) )
print("F1_score: " + str(f1_score(Y_dev, Y_predict, average='macro')))