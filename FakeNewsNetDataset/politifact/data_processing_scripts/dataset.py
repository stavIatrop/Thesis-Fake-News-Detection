import pandas as pd
import math
from textblob import TextBlob
import cufflinks as cf
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_politifact.csv", ',')
labels = df['label'].values
fake_per = sum(labels) / len(labels) * 100
print("Fake articles: " + str(sum(labels)))
print("Articles: " + str(len(labels)))
print("Fake articles: " + str((fake_per)) + "%, True articles: " + str((100 - fake_per)) + "%")


df['polarity'] = df['text'].map(lambda text: TextBlob(text).sentiment.polarity)
df.to_csv('testefa.csv',sep=',',index = False ,header = False)

polarity = df['polarity'].values
true = list()
fake = list()

for i in range(len(labels)):

    if labels[i] == 0:
        true.append(polarity[i])
    else:
        fake.append(polarity[i])
    
plt.hist(true, bins=50, edgecolor='black', label='True articles', log=True, alpha = 0.5, color='orange')

plt.hist(fake, bins=50, edgecolor='black', label='Fake articles', log=True, alpha = 0.5, color='blue')

plt.title("Histogram of polarity measurement of articles")
plt.xlabel("Polarity score")
plt.ylabel("Total articles")
plt.grid(True)
plt.legend(loc='upper right')
plt.show()