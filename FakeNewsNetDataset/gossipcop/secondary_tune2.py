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
import numpy as np
from sklearn.model_selection import KFold


#Load train data
X = pd.read_csv("train_gossipcop_vol2.csv", ",")
Y = X['label'].values
X = X['text'].values

#Load dev data
# X_dev = pd.read_csv("dev_gossipcop.csv", ",")
# Y_dev = X_dev['label'].values
# X_dev = X_dev['text'].values

# X_test = pd.read_csv("test_gossipcop.csv", ",")
# Y_test = X_test['label'].values
# X_test = X_test['text'].values


stopwords = set(ENGLISH_STOP_WORDS)
accuracy_knn = np.zeros((5, 71))
f1_knn = np.zeros((5, 71))

accuracy_LR = np.zeros((5, 71))
f1_LR = np.zeros((5, 71))

accuracy_DT = np.zeros((5, 71))
f1_DT = np.zeros((5, 71))

accuracy_RF = np.zeros((5, 71))
f1_RF = np.zeros((5, 71))

accuracy_svm = np.zeros((5, 71))
f1_svm = np.zeros((5, 71))

df = [i * 0.01 for i in range(20, 91)]
split = 0

kf = KFold(n_splits=5,random_state=42, shuffle=True)
for train, test in kf.split(X):
        
    X_train = X[train]
    X_test = X[test]
    Y_train = Y[train]
    Y_test = Y[test]
    
    df_index = 0
    
    for i in df:
        print(i)
        #df.append(i * 0.01)

        vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = i , stop_words=stopwords)
        X_train2 = vectorizer.fit_transform(X_train)
        X_test2 = vectorizer.transform(X_test)

        svd = TruncatedSVD(n_components= 150, algorithm='arpack', random_state=42)
        X_train2 = svd.fit_transform(X_train2)
        X_test2 = svd.transform(X_test2)

        #SVM
        svm = SVC(random_state=42)
        #svm = SVC(C=10, kernel='linear', random_state=42) 
        svm.fit(X_train2,Y_train)
        Y_predict_test = svm.predict(X_test2)
    
        acc = accuracy_score(Y_test, Y_predict_test)
        F1 = f1_score(Y_test, Y_predict_test)

        accuracy_svm[split][df_index] = acc
        f1_svm[split][df_index] = F1

        print(acc)
        print(F1)

        #KNN
        knn = KNeighborsClassifier()
        #knn = KNeighborsClassifier(metric='minkowski', n_neighbors=1, weights='uniform', p = 4)
        knn.fit(X_train2,Y_train)
        Y_predict_test = knn.predict(X_test2)
    
        acc = accuracy_score(Y_test, Y_predict_test)
        F1 = f1_score(Y_test, Y_predict_test)

        accuracy_knn[split][df_index] = acc
        f1_knn[split][df_index] = F1

        print(acc)
        print(F1)

        #LR
        LR =LogisticRegression(random_state=42)
        #LR = LogisticRegression(C = 100, penalty = 'l2', solver = 'liblinear', random_state=1)
        LR.fit(X_train2,Y_train)
        Y_predict_test = LR.predict(X_test2)
    
        acc = accuracy_score(Y_test, Y_predict_test)
        F1 = f1_score(Y_test, Y_predict_test)

        accuracy_LR[split][df_index] = acc
        f1_LR[split][df_index] = F1

        print(acc)
        print(F1)

        #DT
        DT = DecisionTreeClassifier(random_state=42)
        #DT = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, min_samples_split = 70, random_state=1) 
        DT.fit(X_train2,Y_train)
        Y_predict_test = DT.predict(X_test2)
    
        acc = accuracy_score(Y_test, Y_predict_test)
        F1 = f1_score(Y_test, Y_predict_test)

        accuracy_DT[split][df_index] = acc
        f1_DT[split][df_index] = F1

        print(acc)
        print(F1)

        #RF
        RF = RandomForestClassifier(random_state=42)
        #RF = RandomForestClassifier(criterion = 'entropy', max_depth = 6, min_samples_split = 15, n_estimators = 110,random_state=1)
        RF.fit(X_train2,Y_train)
        Y_predict_test = RF.predict(X_test2)
    
        acc = accuracy_score(Y_test, Y_predict_test)
        F1 = f1_score(Y_test, Y_predict_test)

        accuracy_RF[split][df_index] = acc
        f1_RF[split][df_index] = F1

        print(acc)
        print(F1)
        df_index = df_index + 1
    split = split + 1


accuracy_svm = np.sum(accuracy_svm, axis=0)
f1_svm = np.sum(f1_svm, axis=0)
print(accuracy_svm)
print(f1_svm)
accuracy_svm = accuracy_svm / 5
f1_svm = f1_svm / 5
    
print(accuracy_svm)
print(f1_svm)


accuracy_knn = np.sum(accuracy_knn, axis=0)
f1_knn = np.sum(f1_knn, axis=0)
print(accuracy_knn)
print(f1_knn)
accuracy_knn = accuracy_knn / 5
f1_knn = f1_knn / 5

accuracy_LR = np.sum(accuracy_LR, axis=0)
f1_LR = np.sum(f1_LR, axis=0)
print(accuracy_LR)
print(f1_LR)
accuracy_LR = accuracy_LR / 5
f1_LR = f1_LR / 5


accuracy_DT = np.sum(accuracy_DT, axis=0)
f1_DT = np.sum(f1_DT, axis=0)
print(accuracy_DT)
print(f1_DT)
accuracy_DT = accuracy_DT / 5
f1_DT = f1_DT / 5

accuracy_RF = np.sum(accuracy_RF, axis=0)
f1_RF = np.sum(f1_RF, axis=0)
print(accuracy_RF)
print(f1_RF)
accuracy_RF = accuracy_RF / 5
f1_RF = f1_RF / 5



plt.xlabel('Max_df')
plt.ylabel('Score')
plt.title('svm: Scores in relation with max_df')
plt.axis([min(df)-0.05, max(df) + 0.05, min(min(f1_svm), min(accuracy_svm))-0.0001, max(max(f1_svm), max(accuracy_svm))+0.0001])
plt.plot( df, accuracy_svm, 'rx', label = 'Accuracy' )
plt.plot( df, f1_svm, 'gx', label = 'F1-score' )
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig('svm_graph4.png',bbox_inches='tight')


plt.clf()

plt.xlabel('Max_df')
plt.ylabel('Score')
plt.title('LR: Scores in relation with max_df')
plt.axis([min(df)-0.05, max(df) + 0.05, min(min(f1_LR), min(accuracy_LR))-0.0001, max(max(f1_LR), max(accuracy_LR))+0.0001])
plt.plot( df, accuracy_LR, 'rx', label = 'Accuracy' )
plt.plot( df, f1_LR, 'gx', label = 'F1-score' )
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig('LR_graph4.png',bbox_inches='tight')

plt.clf()

plt.xlabel('Max_df')
plt.ylabel('Score')
plt.title('DT: Scores in relation with max_df')
plt.axis([min(df)-0.05, max(df) + 0.05, min(min(f1_DT), min(accuracy_DT))-0.0001, max(max(f1_DT), max(accuracy_DT))+0.0001])
plt.plot( df, accuracy_DT, 'rx', label = 'Accuracy' )
plt.plot( df, f1_DT, 'gx', label = 'F1-score' )
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig('DT_graph4.png',bbox_inches='tight')

plt.clf()

plt.xlabel('Max_df')
plt.ylabel('Score')
plt.title('KNN: Scores in relation with max_df')
plt.axis([min(df)-0.05, max(df) + 0.05, min(min(f1_knn), min(accuracy_knn))-0.0001, max(max(f1_knn), max(accuracy_knn))+0.0001])
plt.plot( df, accuracy_knn, 'rx', label = 'Accuracy' )
plt.plot( df, f1_knn, 'gx', label = 'F1-score' )
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig('KNN_graph4.png',bbox_inches='tight')

plt.clf()

plt.xlabel('Max_df')
plt.ylabel('Score')
plt.title('RF: Scores in relation with max_df')
plt.axis([min(df)-0.05, max(df) + 0.05, min(min(f1_RF), min(accuracy_RF))-0.0001, max(max(f1_RF), max(accuracy_RF))+0.0001])
plt.plot( df, accuracy_RF, 'rx', label = 'Accuracy' )
plt.plot( df, f1_RF, 'gx', label = 'F1-score' )
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig('RF_graph4.png',bbox_inches='tight')


#     svm = SVC(C=10, kernel='linear', random_state=1)

#     knn = KNeighborsClassifier(metric='minkowski', n_neighbors=1, weights='uniform', p = 4)

#     LR = LogisticRegression(C = 100, penalty = 'l2', solver = 'liblinear', random_state=1)          
    
#     DT = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, min_samples_split = 70, random_state=1) 

#     RF = RandomForestClassifier(criterion = 'entropy', max_depth = 2, min_samples_split = 170, n_estimators = 30 ,random_state=1)


#     svm.fit(X_train2, Y_train)
#     Y_predict_dev = svm.predict(X_dev2)

#     acc = accuracy_score(Y_dev, Y_predict_dev)
#     F1 = f1_score(Y_dev, Y_predict_dev)

#     accuracy_svm.append(acc)
#     f1_svm.append(F1)




#     knn.fit(X_train2, Y_train)
#     Y_predict_dev = knn.predict(X_dev2)

#     acc = accuracy_score(Y_dev, Y_predict_dev)
#     F1 = f1_score(Y_dev, Y_predict_dev)

#     accuracy_knn.append(acc)
#     f1_knn.append(F1)




#     LR.fit(X_train2, Y_train)
#     Y_predict_dev = LR.predict(X_dev2)

#     acc = accuracy_score(Y_dev, Y_predict_dev)
#     F1 = f1_score(Y_dev, Y_predict_dev)

#     accuracy_LR.append(acc)
#     f1_LR.append(F1)




#     DT.fit(X_train2, Y_train)
#     Y_predict_dev = DT.predict(X_dev2)

#     acc = accuracy_score(Y_dev, Y_predict_dev)
#     F1 = f1_score(Y_dev, Y_predict_dev)

#     accuracy_DT.append(acc)
#     f1_DT.append(F1)


#     RF.fit(X_train2, Y_train)
#     Y_predict_dev = RF.predict(X_dev2)

#     acc = accuracy_score(Y_dev, Y_predict_dev)
#     F1 = f1_score(Y_dev, Y_predict_dev)

#     accuracy_RF.append(acc)
#     f1_RF.append(F1)





#     print(acc)
#     print(F1)


# plt.xlabel('Max_df')
# plt.ylabel('Score')
# plt.title('svm: Scores in relation with max_df')
# plt.axis([min(df)-0.05, max(df) + 0.05, min(min(f1_svm), min(accuracy_svm))-0.0001, max(max(f1_svm), max(accuracy_svm))+0.0001])
# plt.plot( df, accuracy_svm, 'rx', label = 'Accuracy' )
# plt.plot( df, f1_svm, 'gx', label = 'F1-score' )
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.savefig('svm_graph.png',bbox_inches='tight')

# plt.clf()

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

# plt.clf()

# plt.xlabel('Max_df')
# plt.ylabel('Score')
# plt.title('KNN: Scores in relation with max_df')
# plt.axis([min(df)-0.05, max(df) + 0.05, min(min(f1_knn), min(accuracy_knn))-0.0001, max(max(f1_knn), max(accuracy_knn))+0.0001])
# plt.plot( df, accuracy_knn, 'rx', label = 'Accuracy' )
# plt.plot( df, f1_knn, 'gx', label = 'F1-score' )
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.savefig('KNN_graph.png',bbox_inches='tight')

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
