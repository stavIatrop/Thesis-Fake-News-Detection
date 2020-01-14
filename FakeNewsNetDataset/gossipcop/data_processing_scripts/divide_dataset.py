import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

list_train = [['id', 'text', 'label']]
list_dev = [['id', 'text', 'label']]
list_test = [['id', 'text', 'label']]

X = pd.read_csv("stemmed_gossipcop.csv", ",", usecols=['id', 'text', 'label'])

y = X['label'].values
X = X.values

# X_train, X_rest, Y_train, Y_rest = train_test_split(
#     X, y, test_size=0.4, random_state=42)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_rest = X[train_index], X[test_index]
    Y_train, Y_rest = y[train_index], y[test_index]


# X_dev, X_test, Y_dev, Y_test = train_test_split(
#     X_rest, Y_rest, test_size=0.5, random_state=42)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)
for dev_index, test_index in sss.split(X_rest, Y_rest):
    X_dev, X_test = X_rest[dev_index], X_rest[test_index]
    Y_dev, Y_test = Y_rest[dev_index], Y_rest[test_index]

for i in range(len(Y_train)):

    list_train.append([X_train[i][0], X_train[i][1], Y_train[i]])

for i in range(len(Y_dev)):

    list_dev.append([X_dev[i][0], X_dev[i][1], Y_dev[i]])
    
for i in range(len(Y_test)):

    list_test.append([X_test[i][0], X_test[i][1], Y_test[i]])

print("Lists made.")
df_train = pd.DataFrame(list_train)
df_train.to_csv('train_gossipcop.csv',sep=',',index = False ,header = False)
print("train made.")
df_dev = pd.DataFrame(list_dev)
df_dev.to_csv('dev_gossipcop.csv',sep=',',index = False ,header = False)
print("dev made.")
df_test = pd.DataFrame(list_test)
df_test.to_csv('test_gossipcop.csv',sep=',',index = False ,header = False)
print("test made.")
