import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

X_train = pd.read_csv("train_politifact.csv", ",")
Y_train = X_train['label'].values
X_train = X_train['text'].values
print("Train set read.")

stopwords = set(ENGLISH_STOP_WORDS)

vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.25, stop_words=stopwords)
X_train = vectorizer.fit_transform(X_train)

variance = list()
nc_list = list()
eigen = list()

print(min(X_train.shape))
for nc in range(50, min(X_train.shape), 50):
    print(nc)
    nc_list.append(nc)
    svd = TruncatedSVD(n_components=nc, algorithm='arpack', random_state=42)
    X_train2 = svd.fit_transform(X_train)
    variance.append(svd.explained_variance_ratio_.sum())
    eigen.append(svd.singular_values_.sum())


plt.plot(nc_list, variance, color='b', label='Variance')
plt.xlabel('Number of components')
plt.ylabel('Variance explained')
plt.title('Variance explained regarding number of components')
plt.grid(True)
plt.show()

plt.clf()

plt.plot(range(svd.singular_values_.shape[0]), svd.singular_values_, color='r', label='Eigenvalues')
plt.xlabel('Number of components')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues regarding number of components')
plt.grid(True)
plt.show()