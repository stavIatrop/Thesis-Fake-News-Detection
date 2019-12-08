import pandas as pd
import math
df = pd.read_csv("test_isot.csv", ',', usecols=['label'])
labels = df['label'].values
true_per = sum(labels) / len(labels) * 100
print("True articles: " + str(math.ceil(true_per)) + "% , Fake articles: " + str(math.floor(100 - true_per)) + "%")