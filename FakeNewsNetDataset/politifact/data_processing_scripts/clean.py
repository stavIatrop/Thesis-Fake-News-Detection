import pandas as pd
import re, string


X = pd.read_csv("politifact.csv", ",", encoding = "utf-8")
X_text = X['text'].values



for i in range(len(X_text)):

    words = X_text[i].split()
    filtered_list = []
    for word in words:
        pattern = re.compile('[^\u0000-\u007F]+', re.UNICODE)  #Remove all non-alphanumeric characters
        
        word = pattern.sub('', word)
        word = word.translate(str.maketrans('', '', string.punctuation))
        filtered_list.append(word)
        result = ' '.join(filtered_list)
        
    X_text[i] = result      


list1 = [['id', 'text', 'label']]
for i in range(0, len(X_text)):
      list1.append([X['id'][i], X_text[i], X['label'][i]])
df1 = pd.DataFrame(list1)
df1.to_csv('cleaned_politifact.csv',sep=',',index = False ,header = False)
