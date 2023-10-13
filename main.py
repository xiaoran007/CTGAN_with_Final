import Classifier
import DatasetsLoader
import DataGenerator


Classifier_list = ['MLP_High_Normal',
                   'MLP_Low_Normal',
                   'MLP_High_UnNormal',
                   'MLP_Low_UnNormal'
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

X_train, y_train, X_val, y_val, X_test, y_test = DatasetsLoader.load_PredictTerm_data()
dataobj = DataGenerator.DataGenerator(X_train, y_train, X_val, y_val, X_test, y_test)
dataobj.Set_CTGAN()
X_train, y_train, X_val, y_val, X_test, y_test = dataobj.GenerateCTGANData()
classifier_obj = Classifier.Classifier(X_train, y_train, X_val, y_val, X_test, y_test, classifier='XGBClassifier', model='CTGAN', epoch=30, batch_size=512)
test_accuracy_array, train_accuracy_array, F1_score_binary_array, F1_score_micro_array, test_auc_array, train_auc_array, \
    precision_array, recall_array, cm_list = classifier_obj.Test()
print(test_auc_array)
