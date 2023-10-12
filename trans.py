import pandas as pd

cols = ['Sequence Name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'class']
dataset = pd.read_csv('Datasets/Yeast/yeast.data', names=cols, delimiter='\s+')
print(dataset)
dataset = dataset.dropna()
print(dataset)
dataset.to_csv('Datasets/Yeast/yeast_new.csv', index=False)
