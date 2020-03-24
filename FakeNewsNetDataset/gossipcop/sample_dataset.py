import pandas
import random


# df = pandas.read_csv("gossipcop_final.csv" , ",")
# labels = df['label'].values
# texts = df['text'].values
# ids = df['id'].values

# list_true = [['id', 'text', 'label']]
# list_fake = [['id', 'text', 'label']]
# text_true = []
# ids_true = []
# labels_true = []

# text_fake = []
# ids_fake = []
# labels_fake = []

# for i in range(0, len(ids)):

#     if labels[i] == 1: # fake texts
#         list_fake.append([ids[i], texts[i], labels[i]])
#     else:
#         list_true.append([ids[i], texts[i], labels[i]])

# list_fake = pandas.DataFrame(list_fake)
# list_fake.to_csv('fake_gossipcop_not_stemmed.csv',sep=',',index = False ,header = False)

# list_true = pandas.DataFrame(list_true)
# list_true.to_csv('true_gossipcop_not_stemmed.csv',sep=',',index = False ,header = False)


# filename = "true_gossipcop_not_stemmed.csv"
# n = 13918
# s = 4126 #desired sample size
# skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
# df = pandas.read_csv(filename, skiprows=skip)

# df.to_csv('true_sampled_not_stemmed.csv', header=['id', 'text', 'label'])

df = pandas.read_csv('balanced.csv', header=None)

ds = df.sample(frac=1)

ds.to_csv('balanced_final_not_stemmed.csv')