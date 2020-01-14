import pandas as pd
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
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

svd = TruncatedSVD(n_components=200,algorithm='arpack', random_state=42)
X_train = svd.fit_transform(X_train)
print("SVD performed.")

x = range(1, 20)
train_scores, valid_scores = validation_curve(RandomForestClassifier(), X_train, y_train, "max_depth", x ,   cv=3, verbose=1000, n_jobs=-1, scoring='f1')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("Max_depth")
plt.ylabel("F1 Score")
plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve RandomForest-Max_depth")
plt.legend()

plt.show()

plt.clf()

x =range(2, 502)
train_scores, valid_scores = validation_curve(RandomForestClassifier(), X_train, y_train, "min_samples_split", x,   cv=3, verbose=1000, n_jobs=-1, scoring='f1')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("min_samples_split values")
plt.ylabel("F1 Score")
plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve RF-min_samples_split")
plt.legend()

plt.show()

plt.clf()

x =range(2, 502, 10)
train_scores, valid_scores = validation_curve(RandomForestClassifier(), X_train, y_train, "n_estimators", x,   cv=3, verbose=1000, n_jobs=-1, scoring='f1')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("n_estimators values")
plt.ylabel("F1 Score")
plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve RF-n_estimators")
plt.legend()

plt.show()

# x = [1, 2]
# train_scores, valid_scores = validation_curve(RandomForestClassifier(), X_train, y_train, "n_estimators", x,   cv=3, verbose=1000, n_jobs=-1)

# train_scores = np.mean(train_scores, axis=1)
# valid_scores = np.mean(valid_scores, axis=1)

# plt.plot(x, train_scores, label="Train score")
# plt.plot(x, valid_scores, label = "Validation score")
# plt.grid(True)
# plt.xlabel("n_estimators values")
# plt.ylabel("Score")
# plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
# plt.title("Validation Curve RF-n_estimators")
# plt.legend()

# plt.show()
