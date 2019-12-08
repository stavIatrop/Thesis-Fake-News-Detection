import pandas as pd
import math

df = pd.read_csv("train_gossipcop.csv", ',', usecols=['label'])
labels = df['label'].values
fake_per = sum(labels) / len(labels) * 100
print("Articles: " + str(len(labels)))
print("Fake articles: " + str((fake_per)) + "%, True articles: " + str((100 - fake_per)) + "%")