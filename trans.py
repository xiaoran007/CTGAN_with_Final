import pandas as pd

cols = ['height', 'length', 'area', 'eccen', 'p_black', 'p_and', 'mean_tr', 'blackpix', 'blackand',
                            'wb_trans', 'class']
dataset = pd.read_csv('Datasets/Predict Term Deposit/Assignment-2_Data.csv', delimiter=',')
print(dataset)
dataset = dataset.dropna()
print(dataset)
dataset.to_csv('Datasets/Predict Term Deposit/Assignment-2_Data_dropna.csv', index=False)
