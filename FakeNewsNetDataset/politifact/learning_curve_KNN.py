import pandas as pd
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt

X_train = pd.read_csv("train_politifact_vol2.csv", ",", usecols=['text', 'label'])
y_train = X_train['label'].values.flatten()
X_train = X_train['text'].values.flatten()
print("Read")

stopwords = set(ENGLISH_STOP_WORDS)
vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.53, stop_words=stopwords)
X_train = vectorizer.fit_transform(X_train)
print("Vectorized.")

svd = TruncatedSVD(n_components=50,algorithm='arpack', random_state=42)
X_train = svd.fit_transform(X_train)
print("SVD performed.")

# x =range(1, 20)
# train_scores, valid_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, "n_neighbors", x,   cv=5, verbose=1000, n_jobs=-1, scoring='f1')

# train_scores = np.mean(train_scores, axis=1)
# valid_scores = np.mean(valid_scores, axis=1)

# plt.plot(x, train_scores, label="Train score")
# plt.plot(x, valid_scores, label = "Validation score")
# plt.grid(True)
# plt.xlabel("n_neighbors values")
# plt.ylabel("f1 Score")
# plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
# plt.title("Validation Curve KNN-n_neighbors")
# plt.legend()

# plt.show()

# plt.clf()

# x = [1, 2]
# train_scores, valid_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, "weights", ['uniform', 'distance'], cv=5, verbose=1000, n_jobs=-1, scoring='f1')

# train_scores = np.mean(train_scores, axis=1)
# valid_scores = np.mean(valid_scores, axis=1)

# plt.plot(x, train_scores, label="Train score")
# plt.plot(x, valid_scores, label = "Validation score")
# plt.grid(True)
# plt.xlabel("weights values")
# plt.ylabel("f1 Score")
# plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)) - 0.1 , max(max(train_scores), max(valid_scores)) + 0.1])
# plt.title("Validation Curve KNN-weights")
# plt.legend()

# plt.show()

# plt.clf()

# x = [1, 2]
# train_scores, valid_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, "metric", ['euclidean', 'manhattan'], cv=5, verbose=1000, n_jobs=-1, scoring='f1')

# train_scores = np.mean(train_scores, axis=1)
# valid_scores = np.mean(valid_scores, axis=1)

# plt.plot(x, train_scores, label="Train score")
# plt.plot(x, valid_scores, label = "Validation score")
# plt.grid(True)
# plt.xlabel("metric values")
# plt.ylabel("f1 Score")
# plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)) - 0.1 , max(max(train_scores), max(valid_scores)) + 0.1])
# plt.title("Validation Curve KNN-metric")
# plt.legend()

# plt.show()

# plt.clf()

# x = range(3, 11)
# train_scores, valid_scores = validation_curve(KNeighborsClassifier( metric='minkowski'), X_train, y_train, "p", x, cv=5, verbose=1000, n_jobs=-1, scoring='f1')

# train_scores = np.mean(train_scores, axis=1)
# valid_scores = np.mean(valid_scores, axis=1)

# plt.plot(x, train_scores, label="Train score")
# plt.plot(x, valid_scores, label = "Validation score")
# plt.grid(True)
# plt.xlabel("p value for 'minkowski' metric values")
# plt.ylabel("f1 Score")
# plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)) - 0.1 , max(max(train_scores), max(valid_scores)) + 0.1])
# plt.title("Validation Curve KNN-p value for 'minkowski' metric")
# plt.legend()

# plt.show()

# plt.clf()


# x =range(1, 20)
# train_scores, valid_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, "n_neighbors", x,   cv=5, verbose=1000, n_jobs=-1, scoring='accuracy')

# train_scores = np.mean(train_scores, axis=1)
# valid_scores = np.mean(valid_scores, axis=1)

# plt.plot(x, train_scores, label="Train score")
# plt.plot(x, valid_scores, label = "Validation score")
# plt.grid(True)
# plt.xlabel("n_neighbors values")
# plt.ylabel("accuracy Score")
# plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)), max(max(train_scores), max(valid_scores))])
# plt.title("Validation Curve KNN-n_neighbors")
# plt.legend()

# plt.show()

# plt.clf()

# x = [1, 2]
# train_scores, valid_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, "weights", ['uniform', 'distance'], cv=5, verbose=1000, n_jobs=-1, scoring='accuracy')

# train_scores = np.mean(train_scores, axis=1)
# valid_scores = np.mean(valid_scores, axis=1)

# plt.plot(x, train_scores, label="Train score")
# plt.plot(x, valid_scores, label = "Validation score")
# plt.grid(True)
# plt.xlabel("weights values")
# plt.ylabel("accuracy Score")
# plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)) - 0.1 , max(max(train_scores), max(valid_scores)) + 0.1])
# plt.title("Validation Curve KNN-weights")
# plt.legend()

# plt.show()

# plt.clf()

# x = [1, 2]
# train_scores, valid_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, "metric", ['euclidean', 'manhattan'], cv=5, verbose=1000, n_jobs=-1, scoring='accuracy')

# train_scores = np.mean(train_scores, axis=1)
# valid_scores = np.mean(valid_scores, axis=1)

# plt.plot(x, train_scores, label="Train score")
# plt.plot(x, valid_scores, label = "Validation score")
# plt.grid(True)
# plt.xlabel("metric values")
# plt.ylabel("accuracy Score")
# plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)) - 0.1 , max(max(train_scores), max(valid_scores)) + 0.1])
# plt.title("Validation Curve KNN-metric")
# plt.legend()

# plt.show()

# plt.clf()

# x = range(3, 11)
# train_scores, valid_scores = validation_curve(KNeighborsClassifier( metric='minkowski'), X_train, y_train, "p", x, cv=5, verbose=1000, n_jobs=-1, scoring='accuracy')

# train_scores = np.mean(train_scores, axis=1)
# valid_scores = np.mean(valid_scores, axis=1)

# plt.plot(x, train_scores, label="Train score")
# plt.plot(x, valid_scores, label = "Validation score")
# plt.grid(True)
# plt.xlabel("p value for 'minkowski' metric values")
# plt.ylabel("accuracy Score")
# plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)) - 0.1 , max(max(train_scores), max(valid_scores)) + 0.1])
# plt.title("Validation Curve KNN-p value for 'minkowski' metric")
# plt.legend()

# plt.show()

# plt.clf()

x = [1, 2, 3, 4]
train_scores, valid_scores = validation_curve(KNeighborsClassifier(metric='minkowski', p=6), X_train, y_train, "algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'], cv=5, verbose=1000, n_jobs=-1, scoring='f1')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("Algorithm values")
plt.ylabel("F1 Score")
plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)) - 0.1 , max(max(train_scores), max(valid_scores)) + 0.1])
plt.title("Validation Curve KNN-Algorithm")
plt.legend()

plt.show()

x = [1, 2, 3, 4]
train_scores, valid_scores = validation_curve(KNeighborsClassifier(metric='minkowski', p=6), X_train, y_train, "algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'], cv=5, verbose=1000, n_jobs=-1, scoring='accuracy')

train_scores = np.mean(train_scores, axis=1)
valid_scores = np.mean(valid_scores, axis=1)

plt.plot(x, train_scores, label="Train score")
plt.plot(x, valid_scores, label = "Validation score")
plt.grid(True)
plt.xlabel("Algorithm values")
plt.ylabel("accuracy Score")
plt.axis([0, max(x) + 1, min(min(train_scores), min(valid_scores)) - 0.1 , max(max(train_scores), max(valid_scores)) + 0.1])
plt.title("Validation Curve KNN-Algorithm")
plt.legend()

plt.show()