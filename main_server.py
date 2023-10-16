import Classifier
import DatasetsLoader
import DataGenerator
import Evaluator
import pandas as pd
import os
import time


Classifier_list = ['MLP_High_Normal',
                   'MLP_Low_Normal',
                   'MLP_High_UnNormal',
                   'MLP_Low_UnNormal',
                   'GaussianNB',
                   'KNeighborsClassifier',
                   'RandomForestClassifier',
                   'DecisionTreeClassifier',
                   'XGBClassifier',
                   'LGBMClassifier',
                   'CatBoostClassifier',
                   'HistGradientBoostingClassifier',
                   'GradientBoostingClassifier',
                   'AdaBoostClassifier',
                   'LogisticRegression']

tf_based_classifier = ['MLP_High_Normal',
                       'MLP_Low_Normal',
                       'MLP_High_UnNormal',
                       'MLP_Low_UnNormal']

sklearn_based_classifier = ['GaussianNB',
                            'KNeighborsClassifier',
                            'RandomForestClassifier',
                            'DecisionTreeClassifier',
                            'XGBClassifier',
                            'LGBMClassifier',
                            'CatBoostClassifier',
                            'HistGradientBoostingClassifier',
                            'GradientBoostingClassifier',
                            'AdaBoostClassifier',
                            'LogisticRegression']

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

Model_list = ['Normal',
              'SMOTE',
              'GAN',
              'SMOTE-GAN',
              'CTGAN']


def SaveCSV(dataframe, classifier, dataset):
    try:
        if not os.path.exists(f'results_server/{classifier}'):
            os.makedirs(f'results_server/{classifier}')
        dataframe.to_csv(f'results_server/{classifier}/{dataset}.csv', index=False)
        return f'Time {time.asctime()} Write results/{classifier}/{dataset} success\n'
    except IOError as e:
        return f'Time {time.asctime()} Err. {e} results/{classifier}/{dataset}\n'


def main(classifier_list, datasets_list):
    with open("log_server.log", 'a') as log:
        status = f'\n------\nTime {time.asctime()} Classifier:{len(classifier_list)} Datasets:{len(datasets_list)} Main start\n'
        log.write(status)
    for classifier in classifier_list:
        with open("log_server.log", 'a') as log:
            status = f'Time {time.asctime()} Classifier:{classifier} start\n'
            log.write(status)
        for dataset in datasets_list:
            try:
                eva_obj = Evaluator.Evaluator(dataset_name=dataset, classifier_name=classifier, rounds=30)
                result_df = eva_obj.evaluate()
                status = SaveCSV(dataframe=result_df, classifier=classifier, dataset=dataset)
                with open("log_server.log", 'a') as log:
                    log.write(status)
            except Exception as e:
                with open("log_server.log", 'a') as log:
                    status = f'Time {time.asctime()} Err. {e} in {classifier}-{dataset}\n'
                    log.write(status)


clist_test = ['LogisticRegression']
dlist_test = ['Africa']

main(classifier_list=clist_test, datasets_list=Datasets_list)

