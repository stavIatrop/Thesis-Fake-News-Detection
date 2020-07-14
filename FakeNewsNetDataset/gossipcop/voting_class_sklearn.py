import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


#Load train data
X_origin = pd.read_csv("train_gossipcop_vol2.csv", ",")
Y_train = X_origin['label'].values
X_origin = X_origin['text'].values
print("Train set read.")
#Load test data
X_test_origin = pd.read_csv("test_gossipcop_vol2.csv", ",")
Y_test = X_test_origin['label'].values
X_test_origin = X_test_origin['text'].values
print("Test set read.")

stopwords = set(ENGLISH_STOP_WORDS)

print("Training SVM...")
vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.25, stop_words=stopwords)
X_train = vectorizer.fit_transform(X_origin)
X_test = vectorizer.transform(X_test_origin)
print("Vectorized.")

svd = TruncatedSVD(n_components=150, algorithm='arpack', random_state=42)
print("SVD prepared.")
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)


print("SVD finished.")


svm = SVC(C=10, gamma='scale', kernel='rbf', random_state=42, probability=True)

svm.fit(X_train, Y_train)
Y_predict_svm = svm.predict(X_test)

print("Training LR...")


LR = LogisticRegression(C=100, penalty='l1', solver='liblinear', max_iter=1000, random_state=42)

LR.fit(X_train, Y_train)
Y_predict_LR = LR.predict(X_test)

print("Training DT...")

DT = DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=420, random_state=42)

DT.fit(X_train, Y_train)
Y_predict_DT = DT.predict(X_test)

print("Training KNN...")

KNN = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')

KNN.fit(X_train, Y_train)
Y_predict_KNN = KNN.predict(X_test)

print("Training RF...")

RF = RandomForestClassifier(criterion='gini', max_depth=None, min_samples_split=2, n_estimators=180, random_state=42) 

RF.fit(X_train, Y_train)
Y_predict_RF = RF.predict(X_test)

#check for agreement on predictions between the classifiers
sum = 0
for i in range(0, len(Y_test)):

    if (Y_predict_svm[i] == Y_predict_RF[i] and Y_predict_RF[i] == Y_predict_KNN[i] and Y_predict_KNN[i] == Y_predict_DT[i] and Y_predict_DT[i] == Y_predict_LR[i] ):
        sum = sum + 1

perc_agree = (sum / len(Y_test)) * 100
print( " Classifiers agree at about: " +  str(perc_agree))

#voting classifier with soft voting using the weights that came up from the ensemble classifier
VC = VotingClassifier(estimators=[('svm', svm), ('KNN', KNN), ('LR', LR), ('DT', DT), ('RF', RF)], voting='soft', weights=[2.75408065, 8.42029977, 0.43180211, 0.16912782, 4.43205492])
VC = VC.fit(X_train, Y_train)
print("Trained.")

Y_predict_test = VC._predict(X_test)
print("test predicted.")

Y_predict_test_final = VC.predict(X_test)
mislabel = 0
_all = len(Y_predict_test)
for i in range(len(Y_predict_test)):

    
    if Y_predict_test_final[i] != Y_test[i]:
        if Y_test[i] in Y_predict_test[i]:
            mislabel = mislabel + 1 

percentage = mislabel / _all
print("Percentage of mislabeled samples that one or more classifier had predicted right on test set:" + 
        str(percentage))

print("test accuracy: " + str(accuracy_score(Y_test, Y_predict_test_final)))
print("test F1 score: " + str(f1_score(Y_test, Y_predict_test_final)))

cf_matrix = confusion_matrix(Y_test, Y_predict_test_final, labels=[0, 1])
htmp_test = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
plt.title("Voting Classifier: Confusion Matrix of Test set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

plt.clf()

#voting classifier with majority voting
VC_hard = VotingClassifier(estimators=[('svm', svm), ('KNN', KNN), ('LR', LR), ('DT', DT), ('RF', RF)], voting='hard')
VC_hard = VC_hard.fit(X_train, Y_train)
print("Trained.")

Y_predict_test = VC_hard._predict(X_test)
print("test predicted.")

Y_predict_test_final = VC_hard.predict(X_test)
mislabel = 0
_all = len(Y_predict_test)
for i in range(len(Y_predict_test)):

    
    if Y_predict_test_final[i] != Y_test[i]:
        if Y_test[i] in Y_predict_test[i]:
            mislabel = mislabel + 1 

percentage = mislabel / _all
print("Percentage of mislabeled samples that one or more classifier had predicted right on test set:" + 
        str(percentage))

print("test accuracy: " + str(accuracy_score(Y_test, Y_predict_test_final)))
print("test F1 score: " + str(f1_score(Y_test, Y_predict_test_final)))

cf_matrix = confusion_matrix(Y_test, Y_predict_test_final, labels=[0, 1])
htmp_test = sns.heatmap(cf_matrix, cmap='Reds', annot=True, fmt='g')
plt.title("Voting Classifier with majority voting: Confusion Matrix of Test set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

plt.clf()