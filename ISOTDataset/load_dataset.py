import pandas as pd

def load_dataset(filename, delimiter):

    X = pd.read_csv(filename, delimiter=delimiter, usecols=['text', 'label'])
    Y = X['label'].values
    X = X['text'].values
    return X, Y

