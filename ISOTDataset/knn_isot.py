import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.parsing.porter import PorterStemmer
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np

X = pd.read_csv("train_isot.csv", ",", usecols=['text', 'label'])
Y_train = X['label'].values
X_train = X['text'].values

X = pd.read_csv("dev_isot.csv", ",", usecols=['text', 'label'])
Y_dev = X['label'].values
X_dev = X['text'].values

#Stopwords removal

stopwords = set(ENGLISH_STOP_WORDS)


for i in np.arange(20, 91):
    print(i * 0.01)
    vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = i * 0.01, stop_words=stopwords)

    X_train2 = vectorizer.fit_transform(X_train) 
    X_dev2 = vectorizer.transform(X_dev) 

    svd = TruncatedSVD(n_components=1000,algorithm='randomized',random_state = 42)
    X_train2 = svd.fit_transform(X_train2)
    X_dev2 = svd.transform(X_dev2)

    clf = KNeighborsClassifier(n_neighbors= 3, weights= 'distance' , metric= 'euclidean')

    clf.fit(X_train2, Y_train)

    print("Trained.")
    Y_predict = clf.predict(X_dev2)


    print("Precision: " + str(precision_score(Y_dev, Y_predict, average='macro')))
    print("Recall: " + str(recall_score(Y_dev, Y_predict, average='macro')))
    print("Accuracy: " + str(accuracy_score(Y_dev, Y_predict)) )
    print( "F1 score: " + str(f1_score(Y_dev, Y_predict, average='macro')))