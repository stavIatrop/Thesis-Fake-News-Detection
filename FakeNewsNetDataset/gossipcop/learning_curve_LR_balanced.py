import pandas as pd
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt

X_train = pd.read_csv("train_gossipcop_balanced.csv", ",", usecols=['text', 'label'])
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

x = [0.1, 1, 10 , 100, 1000, 1100, 1300, 1500]
train_scores, valid_scores = validation_curve(LogisticRegression(random_state=42), X_train, y_train, "C", x ,   cv=3 , verbose=1000, n_jobs=-1, scoring='accuracy')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("C values")
plt.ylabel("accuracy Score")
plt.axis([0, max(x) + 0.001, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve LR-C")
plt.legend()

plt.show()

plt.clf()

x = [1, 2, 3, 4, 5]
train_scores, valid_scores = validation_curve(LogisticRegression(random_state=42), X_train, y_train, "solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] ,   cv=3 , verbose=1000, n_jobs=-1, scoring='accuracy')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("Solver")
plt.ylabel("accuracy Score")
plt.axis([0, max(x) + 0.001, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve LR-Solver")
plt.legend()

plt.show()

plt.clf()


# x = [0.1, 0.3, 0.5, 0.7, 0.9]
# train_scores, valid_scores = validation_curve(LogisticRegression(C=10, solver='saga', penalty='elasticnet'), X_train, y_train, "l1_ratio", x ,   cv=3 , verbose=1000, n_jobs=-1, scoring='f1')

# train_scores = np.mean(train_scores, axis=1)
# valid_scores = np.mean(valid_scores, axis=1)

# plt.plot(x, train_scores, label="Train score")
# plt.plot(x, valid_scores, label = "Validation score")
# plt.grid(True)
# plt.xlabel("L1_ratio(penalty-elasticnet )values")
# plt.ylabel("F1 Score")
# plt.axis([0, max(x) + 0.001, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
# plt.title("Validation Curve LR-L1_ratio(penalty-elasticnet)")
# plt.legend()

# plt.show()


# x = range(10, 5000)
# train_scores, valid_scores = validation_curve(LogisticRegression(C=10, solver='saga'), X_train, y_train, "max_iter", x ,   cv=3 , verbose=1000, n_jobs=-1, scoring='f1')

# train_scores = np.mean(train_scores, axis=1)
# valid_scores = np.mean(valid_scores, axis=1)

# plt.plot(x, train_scores, label="Train score")
# plt.plot(x, valid_scores, label = "Validation score")
# plt.grid(True)
# plt.xlabel("Max_iter values")
# plt.ylabel("F1 Score")
# plt.axis([0, max(x) + 0.001, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
# plt.title("Validation Curve LR-Max_iter")
# plt.legend()

# plt.show()


x = [0.1, 1, 10 , 100, 1000, 1100, 1300, 1500]
train_scores, valid_scores = validation_curve(LogisticRegression(random_state=42), X_train, y_train, "C", x ,   cv=3 , verbose=1000, n_jobs=-1, scoring='f1')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("C values")
plt.ylabel("F1 Score")
plt.axis([0, max(x) + 0.001, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve LR-C")
plt.legend()

plt.show()

plt.clf()

x = [1, 2, 3, 4, 5]
train_scores, valid_scores = validation_curve(LogisticRegression(random_state=42), X_train, y_train, "solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] ,   cv=3 , verbose=1000, n_jobs=-1, scoring='f1')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("Solver")
plt.ylabel("F1 Score")
plt.axis([0, max(x) + 0.001, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
plt.title("Validation Curve LR-Solver")
plt.legend()

plt.show()

plt.clf()