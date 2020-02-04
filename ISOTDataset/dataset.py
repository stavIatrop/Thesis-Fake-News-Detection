import pandas as pd
import math
from textblob import TextBlob
import cufflinks as cf
import matplotlib.pyplot as plt

df = pd.read_csv("isot_rev.csv", ',')
labels = df['label'].values
fake_per = sum(labels) / len(labels) * 100
print("Fake articles: " + str(sum(labels)))
print("Articles: " + str(len(labels)))
print("Fake articles: " + str((fake_per)) + "%, True articles: " + str((100 - fake_per)) + "%")



df['polarity'] = df['text'].map(lambda text: TextBlob(text).sentiment.polarity)
df['text_length'] = df['text'].astype(str).apply(len)
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df.to_csv('testefa.csv',sep=',',index = False ,header = False)

polarity = df['polarity'].values
text_length = df['text_length'].values
wc = df['word_count'].values
true = list()
fake = list()

for i in range(len(labels)):

    if labels[i] == 0:
        true.append(polarity[i])
    else:
        fake.append(polarity[i])
    


plt.hist(fake, bins=50, edgecolor='black', label='Fake articles',  alpha = 0.5, color='blue')
plt.hist(true, bins=50, edgecolor='black', label='True articles',  alpha = 0.5, color='orange')

plt.title("Histogram of polarity measurement of articles")
plt.xlabel("Polarity score")
plt.ylabel("Total articles")
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

plt.clf()

true = list()
fake = list()

for i in range(len(labels)):

    if labels[i] == 0:
        true.append(text_length[i])
    else:
        fake.append(text_length[i])
    

plt.hist(fake, bins=50, edgecolor='black', label='Fake articles',  alpha = 0.5, color='blue')
plt.hist(true, bins=50, edgecolor='black', label='True articles',  alpha = 0.5, color='orange')

plt.title("Histogram of text length of articles")
plt.xlabel("Text Length")
plt.ylabel("Total articles")
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

plt.clf()

true = list()
fake = list()

for i in range(len(labels)):

    if labels[i] == 0:
        true.append(wc[i])
    else:
        fake.append(wc[i])
    

plt.hist(fake, bins=50, edgecolor='black', label='Fake articles',  alpha = 0.5, color='blue')
plt.hist(true, bins=50, edgecolor='black', label='True articles',  alpha = 0.5, color='orange')

plt.title("Histogram of word count of articles")
plt.xlabel("Word count")
plt.ylabel("Total articles")
plt.grid(True)
plt.legend(loc='upper right')
plt.show()
