import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

Datasets_list = ['Africa',
                 'BankNote',
                 'CreditApproval',
                 'CreditRisk',
                 'CreditCard',
                 'Ecoli',
                 'FakeBills',
                 'LoanPrediction',
                 'PageBlock',
                 'PageBlockDel',
                 'PredictTerm',
                 'SouthGermanCredit',
                 'Wine',
                 'WineRed',
                 'Yeast',
                 'YeastUn']


def preprocess(dataset_name):
    if dataset_name == 'Africa':
        return load_Africa_data()
    elif dataset_name == "BankNote":
        return load_BankNote_data()
    elif dataset_name == "CreditApproval":
        return load_CreditApproval_data()
    elif dataset_name == "CreditRisk":
        return load_CreditRisk_data()
    elif dataset_name == "CreditCard":
        return load_CreditCard_data()
    elif dataset_name == "Ecoli":
        return load_Ecoli_data()
    elif dataset_name == "FakeBills":
        return load_FakeBills_data()
    elif dataset_name == "LoanPrediction":
        return load_LoanPrediction_data()
    elif dataset_name == "PageBlock":
        return load_PageBlock_data()
    elif dataset_name == "PageBlockDel":
        return load_PageBlockDel_data()
    elif dataset_name == "PredictTerm":
        return load_PredictTerm_data()
    elif dataset_name == "SouthGermanCredit":
        return load_SouthGermanCredit_data()
    elif dataset_name == "Wine":
        return load_Wine_data()
    elif dataset_name == "WineRed":
        return load_WineRed_data()
    elif dataset_name == "Yeast":
        return load_Yeast_data()
    elif dataset_name == "FakeBills":
        return load_YeastUn_data()


def load_Africa_data():
    dataset = pd.read_csv("Datasets/Africa Economic, Banking and Systemic Crisis Data/african_crises.csv")

    X = dataset.drop(['banking_crisis', 'country'], axis=1)
    y = dataset['banking_crisis']
    y = y.replace({'no_crisis': 0, 'crisis': 1})
    y.astype('int')

    le = LabelEncoder()
    X['cc3'] = le.fit_transform(X['cc3'])
    X = MinMaxScaler().fit_transform(X)

    # Split dataset into 7:1:2 for training : validation : testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=0,
                                                      stratify=y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_BankNote_data():
    dataset = pd.read_csv("Datasets/Bank Note Authentication UCI data/BankNote_Authentication.csv")

    X = dataset.drop(['class'], axis=1)
    y = dataset['class']
    y.astype('int')

    X = MinMaxScaler().fit_transform(X)

    # Split dataset into 7:1:2 for training : validation : testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=0,
                                                      stratify=y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_CreditApproval_data():
    cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class']
    dataset = pd.read_csv('Datasets/Credit Approval/CreditApprovalDataSet.data', names=cols, delimiter=',')
    dataset = dataset.replace('?', np.nan)
    dataset = dataset.dropna()
    X = dataset.drop(['class'], axis=1)
    y = dataset['class']
    y = y.replace({'-': 0, '+': 1})
    y.astype('int')

    le = LabelEncoder()
    X['A1'] = le.fit_transform(X['A1'])
    X['A4'] = le.fit_transform(X['A4'])
    X['A5'] = le.fit_transform(X['A5'])
    X['A6'] = le.fit_transform(X['A6'])
    X['A7'] = le.fit_transform(X['A7'])
    X['A9'] = le.fit_transform(X['A9'])
    X['A10'] = le.fit_transform(X['A10'])
    X['A12'] = le.fit_transform(X['A12'])
    X['A13'] = le.fit_transform(X['A13'])

    X['A2'] = X['A2'].astype(float)
    X['A14'] = X['A14'].astype(float)
    X = MinMaxScaler().fit_transform(X)

    # Split dataset into 7:1:2 for training : validation : testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=0,
                                                      stratify=y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_CreditRisk_data():
    dataset = pd.read_csv("Datasets/Credit Risk/customer_data_drop_na.csv", delimiter=',')
    X = dataset.drop(['label', 'ID'], axis=1)
    y = dataset['label']
    y.astype('int')
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=0,
                                                      stratify=y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_CreditCard_data():
    # dataset = pd.read_excel("datasets/default_of_credit_card_clients.xls", header=1)
    dataset = pd.read_csv("Datasets/Default of Credit Card Clients Dataset/UCI_Credit_Card.csv", delimiter=',')
    X = dataset.drop(['default payment next month', 'ID'], axis=1)
    y = dataset['default payment next month']
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=0,
                                                      stratify=y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_Ecoli_data():
    dataset = pd.read_csv('Datasets/Ecoli4/ecoli4_new.csv', names=['mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class'],
                          delimiter=',')

    X = dataset.drop(['class'], axis=1)
    y = dataset['class']
    y = y.replace({'negative': 0, 'positive': 1})
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=0,
                                                      stratify=y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_FakeBills_data():
    # Column names
    cols = ['Class', 'Diagonal', 'Height_left', 'Height_right', 'Margin_low', 'Margin_up', 'Length']
    dataset = pd.read_csv("Datasets/Fake Bills/fake_bills_drop_na.csv", header=None, names=cols)
    dataset = dataset.replace({True: 0, False: 1})

    X = dataset.drop(['Class'], axis=1)
    y = dataset['Class'].astype("int")

    # Scale the datasets
    X = MinMaxScaler().fit_transform(X)

    # Split dataset into 7:1:2 for training : validation : testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=0,
                                                      stratify=y_train)

    # print("Train Fraud: ", len(y_train[y_train == 1]))
    # print("Train Non-fraud: ", len(y_train[y_train == 0]))

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_LoanPrediction_data():
    dataset = pd.read_csv("Datasets/Loan Prediction Problem Dataset/train_dropna.csv", delimiter=',')
    pass


def load_PageBlock_data():
    pass


def load_PageBlockDel_data():
    pass


def load_PredictTerm_data():
    pass


def load_SouthGermanCredit_data():
    dataset = pd.read_csv("Datasets/South German Credit/SouthGermanCredit.asc", sep=' ')

    X = dataset.drop(['kredit'], axis=1)
    y = dataset['kredit'].astype("int")
    y = y.replace({1: 0, 0: 1})

    X = MinMaxScaler().fit_transform(X)

    # Split dataset into 7:1:2 for training : validation : testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=0,
                                                      stratify=y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_Wine_data():
    pass


def load_WineRed_data():
    pass


def load_Yeast_data():
    pass


def load_YeastUn_data():
    pass


def Check(dataset_name):
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess(dataset_name=dataset_name)
        print(X_train)
        print(y_train)
        print(X_test)
        print(y_test)
    except:
        print("Err")


if __name__ == "__main__":
    for i in Datasets_list:
        print(f"Dataset {i}:")
        Check(i)
        print('\n')


