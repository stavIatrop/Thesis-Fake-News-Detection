import pandas as pd
from gensim.parsing.porter import PorterStemmer
import re, string


X = pd.read_csv("isot_rev.csv", ",")
X_text = X['text'].values
X_title = X['title'].values

p = PorterStemmer()

for i in range(len(X_text)):
    
    words = X_title[i].split()
    filtered_list = []
    for word in words:
        pattern = re.compile('[\W_]+', re.UNICODE)  #Remove all non-alphanumeric characters
        word = pattern.sub('', word)
        word = word.translate(str.maketrans('', '', string.punctuation))
        filtered_list.append(word)
        result = ' '.join(filtered_list)

    X_title[i] = result

    words = X_text[i].split()
    filtered_list = []
    for word in words:
        pattern = re.compile('[\W_]+', re.UNICODE)  #Remove all non-alphanumeric characters
        word = pattern.sub('', word)
        word = word.translate(str.maketrans('', '', string.punctuation))
        filtered_list.append(word)
        result = ' '.join(filtered_list)
        
    X_text[i] = result      #if there is no text, use title
    
    X_title[i] = p.stem_sentence(X_title[i])
    X_text[i] = p.stem_sentence(X_text[i])


list1 = [['id','title', 'text', 'subject', 'date', 'label']]
for i in range(0, len(X_text)):
      list1.append([X['id'][i], X_title[i], X_text[i], X['subject'][i], X['date'][i], X['label'][i]])
df1 = pd.DataFrame(list1)
df1.to_csv('stemmed_ISOT_dataset.csv',sep=',',index = False ,header = False)
