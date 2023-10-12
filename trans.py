import pandas as pd

dataset = pd.read_csv("Datasets/Loan Prediction Problem Dataset/test_Y3wMUE5_7gLdaTN.csv", delimiter=',')
print(dataset)
dataset = dataset.dropna()
print(dataset)
dataset.to_csv('Datasets/Loan Prediction Problem Dataset/test_dropna.csv', index=False)
