from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud
import pandas as pd
from os import path

d = path.dirname(__file__)

X = pd.read_csv("gossipcop_final.csv", ",", encoding = "utf-8")
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