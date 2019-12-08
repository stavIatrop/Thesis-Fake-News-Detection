import pandas as pd
from sklearn.model_selection import train_test_split

list_train = [['id', 'text', 'label']]
list_dev = [['id', 'text', 'label']]
list_test = [['id', 'text', 'label']]

X = pd.read_csv("stemmed_gossipcop.csv", ",", usecols=['id', 'text', 'label'])

y = X['label'].values

X_train, X_rest, Y_train, Y_rest = train_test_split(
    X, y, test_size=0.4, random_state=42)

X_dev, X_test, Y_dev, Y_test = train_test_split(
    X_rest, Y_rest, test_size=0.2, random_state=42)


for i in range(len(Y_train)):

    list_train.append([X_train.iloc[i, X_train.columns.get_loc('id') ], X_train.iloc[i, X_train.columns.get_loc('text') ], Y_train[i]])

for i in range(len(Y_dev)):

    list_dev.append([X_dev.iloc[0, X_dev.columns.get_loc('id') ], X_dev.iloc[0, X_dev.columns.get_loc('text') ], Y_dev[i]])
    
for i in range(len(Y_test)):

    list_test.append([X_test.iloc[0, X_test.columns.get_loc('id') ], X_test.iloc[0, X_test.columns.get_loc('text') ], Y_test[i]])

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
