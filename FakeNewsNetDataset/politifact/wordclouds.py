from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud
import pandas as pd
from os import path
import re, string

d = path.dirname(__file__)

X = pd.read_csv("cleaned_politifact.csv", ",", encoding = "utf-8")
print("Read")
X_text = X['text'].values
X_label = X['label'].values
X_true_list = []
X_fake_list = []

for i in range(len(X_text)):

    if X_label[i] == 1:
        
        X_fake_list.append(X_text[i])
    else:
        X_true_list.append(X_text[i])
    
X_true_text = '\n'.join(X_true_list)
X_fake_text = '\n'.join(X_fake_list)


stopwords = set(ENGLISH_STOP_WORDS)
stopwords.add("said")
stopwords.add("say")
stopwords.add("says")

wordcloud = WordCloud(stopwords=stopwords, scale=3)

wordcloud.generate(X_true_text)
wordcloud.to_file(path.join(d, "True.png"))
wordcloud.generate(X_fake_text)
wordcloud.to_file(path.join(d, "Fake.png"))
for i in range(len(X_fake_list)):
    
    words = X_fake_list[i].lower().split()
    words = [word for word in words if not word in stopwords]
    
    filtered_list = []
    for word in words:
        
            
        pattern = re.compile('[^\u0000-\u007F]+', re.UNICODE)  #Remove all non-alphanumeric characters
        
        word = pattern.sub('', word)
        word = word.translate(str.maketrans('', '', string.punctuation))
        filtered_list.append(word)
        result = ' '.join(filtered_list)
            
    X_fake_list[i] = result

for i in range(len(X_true_list)):
    
    words = X_true_list[i].lower().split()
    words = [word for word in words if not word in stopwords]
    
    filtered_list = []
    for word in words:
        
            
        pattern = re.compile('[^\u0000-\u007F]+', re.UNICODE)  #Remove all non-alphanumeric characters
        
        word = pattern.sub('', word)
        word = word.translate(str.maketrans('', '', string.punctuation))
        filtered_list.append(word)
        result = ' '.join(filtered_list)
            
    X_true_list[i] = result


from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sb
text_cnt_true = Counter(" ".join(X_true_list).lower().split()).most_common(30)
text_cnt_fake = Counter(" ".join(X_fake_list).lower().split()).most_common(30)

common_words_true = pd.DataFrame(text_cnt_true, columns=['Words', 'Counts'])
common_words_fake = pd.DataFrame(text_cnt_fake, columns=['Words', 'Counts'])

plt.figure(figsize=(15, 10))

sb.barplot(y='Words', x = 'Counts', data=common_words_fake)
plt.show()
plt.clf()
sb.barplot(y='Words', x = 'Counts', data=common_words_true)
plt.show()
plt.clf()