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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#Load train data
X_train = pd.read_csv("train_gossipcop.csv", ",")
Y_train = X_train['label'].values
X_train = X_train['text'].values

#Load dev data
X_dev = pd.read_csv("dev_gossipcop.csv", ",")
Y_dev = X_dev['label'].values
X_dev = X_dev['text'].values

# X_test = pd.read_csv("test_gossipcop.csv", ",")
# Y_test = X_test['label'].values
# X_test = X_test['text'].values


stopwords = set(ENGLISH_STOP_WORDS)
# accuracy_knn = list()
# f1_knn = list()
# accuracy_LR = list()
# f1_LR = list()
accuracy_DT = list()
f1_DT = list()
# accuracy_RF = list()
# f1_RF = list()
# accuracy_svm = list()
# f1_svm = list()
df = list()

for i in range(20, 91, 5):
    print(i * 0.01)
    df.append(i * 0.01)
    vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = i * 0.01, stop_words=stopwords)
    X_train2 = vectorizer.fit_transform(X_train)
    X_dev2 = vectorizer.transform(X_dev)
    
    svd = TruncatedSVD(n_components=1000, algorithm='arpack', random_state=42)
    X_train2 = svd.fit_transform(X_train2)
    X_dev2 = svd.transform(X_dev2)
    
    #svm = SVC(C=10,gamma=1, kernel='rbf')

    #knn = KNeighborsClassifier(metric='minkowski', n_neighbors=5, weights='distance', p = 6)

    #LR = LogisticRegression(C=10, penalty='l2', solver='liblinear')          #best params found regarding accuracy
    #LR = LogisticRegression(C=10, penalty='l1', solver='saga')          #best params found regarding F1_score
    DT = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=400)

    #RF = RandomForestClassifier(n_estimators=300, criterion="gini", max_depth=16, min_samples_split=20)


    # svm.fit(X_train2, Y_train)
    # Y_predict_dev = svm.predict(X_dev2)

    # accuracy_svm.append(accuracy_score(Y_dev, Y_predict_dev))
    # f1_svm.append(f1_score(Y_dev, Y_predict_dev))

    # knn.fit(X_train2, Y_train)
    # Y_predict_dev = knn.predict(X_dev2)

    # accuracy_knn.append(acc)
    # f1_knn.append(F1)

    # LR.fit(X_train2, Y_train)
    # Y_predict_dev = LR.predict(X_dev2)

    

    # accuracy_LR.append(acc)
    # f1_LR.append(F1)

    DT.fit(X_train2, Y_train)
    Y_predict_dev = DT.predict(X_dev2)

    acc = accuracy_score(Y_dev, Y_predict_dev)
    F1 = f1_score(Y_dev, Y_predict_dev)

    accuracy_DT.append(acc)
    f1_DT.append(F1)

    
    
    print(acc)
    print(F1)


# plt.xlabel('Max_df')
# plt.ylabel('Score')
# plt.title('svm: Scores in relation with max_df')
# plt.axis([min(df)-0.05, max(df) + 0.05, min(min(f1_svm), min(accuracy_svm))-0.0001, max(max(f1_svm), max(accuracy_svm))+0.0001])
# plt.plot( df, accuracy_svm, 'rx', label = 'Accuracy' )
# plt.plot( df, f1_svm, 'gx', label = 'F1-score' )
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.savefig('svm_graph.png',bbox_inches='tight')

# plt.xlabel('Max_df')
# plt.ylabel('Score')
# plt.title('LR: Scores in relation with max_df')
# plt.axis([min(df)-0.05, max(df) + 0.05, min(min(f1_LR), min(accuracy_LR))-0.0001, max(max(f1_LR), max(accuracy_LR))+0.0001])
# plt.plot( df, accuracy_LR, 'rx', label = 'Accuracy' )
# plt.plot( df, f1_LR, 'gx', label = 'F1-score' )
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.savefig('LR_graph.png',bbox_inches='tight')

# plt.clf()

# plt.xlabel('Max_df')
# plt.ylabel('Score')
# plt.title('DT: Scores in relation with max_df')
# plt.axis([min(df)-0.05, max(df) + 0.05, min(min(f1_DT), min(accuracy_DT))-0.0001, max(max(f1_DT), max(accuracy_DT))+0.0001])
# plt.plot( df, accuracy_DT, 'rx', label = 'Accuracy' )
# plt.plot( df, f1_DT, 'gx', label = 'F1-score' )
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.savefig('DT_graph.png',bbox_inches='tight')

# # plt.clf()

plt.xlabel('Max_df')
plt.ylabel('Score')
plt.title('KNN: Scores in relation with max_df')
plt.axis([min(df)-0.05, max(df) + 0.05, min(min(f1_knn), min(accuracy_knn))-0.0001, max(max(f1_knn), max(accuracy_knn))+0.0001])
plt.plot( df, accuracy_knn, 'rx', label = 'Accuracy' )
plt.plot( df, f1_knn, 'gx', label = 'F1-score' )
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig('KNN_graph.png',bbox_inches='tight')

# plt.clf()

# plt.xlabel('Max_df')
# plt.ylabel('Score')
# plt.title('RF: Scores in relation with max_df')
# plt.axis([min(df)-0.05, max(df) + 0.05, min(min(f1_RF), min(accuracy_RF))-0.0001, max(max(f1_RF), max(accuracy_RF))+0.0001])
# plt.plot( df, accuracy_RF, 'rx', label = 'Accuracy' )
# plt.plot( df, f1_RF, 'gx', label = 'F1-score' )
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.savefig('RF_graph.png',bbox_inches='tight')
