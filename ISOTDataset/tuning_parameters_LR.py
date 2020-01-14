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
from sklearn.linear_model import LogisticRegression

X = pd.read_csv("train_isot.csv", ",", usecols=['text', 'label'])
y_train = X['label'].values.flatten()
X_train = X['text'].values.flatten()


stopwords = set(ENGLISH_STOP_WORDS)

parameters = [
    {'C': [1, 10, 100, 1000], 'penalty' : ['l2'], 'solver' : ['sag', 'saga'] },
    {'C': [1, 10, 100, 1000], 'penalty' : ['l1'], 'solver' : ['saga'] }   
]

vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.25, stop_words=stopwords)
X_train = vectorizer.fit_transform(X_train)

svd = TruncatedSVD(n_components=1000,algorithm='randomized', random_state=42)
X_train = svd.fit_transform(X_train)

LR = LogisticRegression()
scores = [ 'accuracy', 'f1']

for score in scores:

    print("# Tuning hyper-parameters for %s" % score)
    print()

    if score == 'accuracy':
    
        clf = GridSearchCV(LR, parameters, scoring='%s' % score, cv=5)
    else:

        clf = GridSearchCV(LR, parameters, scoring='%s_macro' % score, cv=5)

    clf.fit(X_train, y_train)


    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)