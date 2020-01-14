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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv("train_isot.csv", ",", usecols=['text', 'label'])
y_train = X_train['label'].values.flatten()
X_train = X_train['text'].values.flatten()

X_dev = pd.read_csv("dev_isot.csv", ",", usecols=['text', 'label'])
y_dev = X_dev['label'].values.flatten()
X_dev = X_dev['text'].values.flatten()

parameters = {
    'n_neighbors' : [3, 5, 11, 19],
    'weights' : ['uniform', 'distance'],
    'metric' : ['euclidean', 'manhattan']
}

stopwords = set(ENGLISH_STOP_WORDS)

vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.26, stop_words=stopwords)
X_train = vectorizer.fit_transform(X_train)
X_dev = vectorizer.transform(X_dev)

svd = TruncatedSVD(n_components=1000,algorithm='randomized', random_state=42)
X_train = svd.fit_transform(X_train)
X_dev = svd.transform(X_dev)

knn = KNeighborsClassifier()
scores = [ 'accuracy', 'f1']

for score in scores:

    print("# Tuning hyper-parameters for %s" % score)
    print()

    if score == 'accuracy':
        clf = GridSearchCV(knn, parameters, scoring='%s' % score, cv=3)
    else:
        clf = GridSearchCV(knn, parameters, scoring='%s_macro' % score, cv=3)

    clf.fit(X_train, y_train)
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    