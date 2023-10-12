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
    pass


def load_BankNote_data():
    pass


def load_CreditApproval_data():
    pass


def load_CreditRisk_data():
    pass


def load_CreditCard_data():
    pass


def load_Ecoli_data():
    pass


def load_FakeBills_data():
    pass


def load_LoanPrediction_data():
    pass


def load_PageBlock_data():
    pass


def load_PageBlockDel_data():
    pass


def load_PredictTerm_data():
    pass


def load_SouthGermanCredit_data():
    pass


def load_Wine_data():
    pass


def load_WineRed_data():
    pass


def load_Yeast_data():
    pass


def load_YeastUn_data():
    pass
