import pandas as pd
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt

X_train = pd.read_csv("train_gossipcop.csv", ",", usecols=['text', 'label'])
y_train = X_train['label'].values.flatten()
X_train = X_train['text'].values.flatten()
print("Read")

stopwords = set(ENGLISH_STOP_WORDS)
vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.25, stop_words=stopwords)
X_train = vectorizer.fit_transform(X_train)
print("Vectorized.")

svd = TruncatedSVD(n_components=250,algorithm='arpack', random_state=42)
X_train = svd.fit_transform(X_train)
print("SVD performed.")

x = np.logspace(start=-7, stop=1, base=10.0, num=9)
train_scores, valid_scores = validation_curve(SVC(), X_train, y_train, "gamma", x,   cv=5, verbose=1000, n_jobs=-1, scoring='f1')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("Gamma values")
plt.ylabel("Score")
plt.axis([0, max(x) + 0.0000001, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve SVM-gamma")
plt.legend()

plt.show()

plt.clf()

x = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_scores, valid_scores = validation_curve(SVC(gamma='auto'), X_train, y_train, "C", x ,   cv=5, verbose=1000, n_jobs=-1, scoring='f1')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("C values")
plt.ylabel("F1 Score")
plt.axis([0, max(x) + 0.001, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve SVM-C")
plt.legend()

plt.show()

plt.clf()

x = [1, 2]
train_scores, valid_scores = validation_curve(SVC(gamma=1), X_train, y_train, "kernel", ['linear', 'rbf'] ,   cv=5, verbose=1000, n_jobs=-1, scoring='f1')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("Kernel")
plt.ylabel("F1 Score")
plt.axis([0, max(x) + 0.001, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve SVM-Kernel")
plt.legend()

plt.show()