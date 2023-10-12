from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import math

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


Classifier_dict = {'GaussianNB': GaussianNB(),
                   'KNeighborsClassifier': KNeighborsClassifier(n_jobs=-1),
                   'RandomForestClassifier': RandomForestClassifier(n_jobs=-1),
                   'DecisionTreeClassifier': DecisionTreeClassifier(),
                   'XGBClassifier': XGBClassifier(tree_method="hist", device="cuda"),
                   'LGBMClassifier': LGBMClassifier(objective='binary'),
                   'CatBoostClassifier': CatBoostClassifier(task_type="GPU", verbose=True),
                   'HistGradientBoostingClassifier': HistGradientBoostingClassifier(verbose=1),
                   'GradientBoostingClassifier': GradientBoostingClassifier(),
                   'AdaBoostClassifier': AdaBoostClassifier(),
                   'LogisticRegression': LogisticRegression(n_jobs=-1)}

pipeline_gnb = Pipeline([('GaussianNB', GaussianNB())])
pipeline_knn = Pipeline([('KNeighborsClassifier', KNeighborsClassifier(n_jobs=-1))])
pipeline_rf = Pipeline([('RandomForestClassifier', RandomForestClassifier(n_jobs=-1))])
pipeline_dt = Pipeline([('DecisionTreeClassifier', DecisionTreeClassifier())])
pipeline_xgb = Pipeline([('XGBClassifier', XGBClassifier(tree_method="hist", device="cuda"))])
pipeline_lgbm = Pipeline([('LGBMClassifier', LGBMClassifier(objective='binary'))])
pipeline_catB = Pipeline([('CatBoostClassifier', CatBoostClassifier(task_type="GPU", verbose=True))])
pipeline_hist = Pipeline([('HistGradientBoostingClassifier', HistGradientBoostingClassifier(verbose=1))])
pipeline_gradB = Pipeline([('GradientBoostingClassifier', GradientBoostingClassifier())])
pipeline_adaB = Pipeline([('AdaBoostClassifier', AdaBoostClassifier())])
pipeline_lr = Pipeline([('LogisticRegression', LogisticRegression(n_jobs=-1))])


class Classifier:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, classifier, model, epoch=30, batch_size=8192):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.classifier = classifier
        self.model = model
        self.batch_size = batch_size
        self.epoch = epoch

    def GetClassifier(self):
        if self.classifier in tf_based_classifier:
            pass
        elif self.classifier in sklearn_based_classifier:
            return Classifier_dict[self.classifier]
        else:
            print("Err.")

    def Test(self):
        if self.classifier in tf_based_classifier:
            pass
        elif self.classifier in sklearn_based_classifier:
            test_accuracy_array = []
            train_accuracy_array = []
            F1_score_binary_array = []
            F1_score_micro_array = []
            test_auc_array = []
            train_auc_array = []
            precision_array = []
            recall_array = []
            cm_list = []
            for i in range(self.epoch):
                classifier = self.GetClassifier()
                classifier.fit(self.X_train, self.y_train)
                y_test_pred_proba = classifier.predict_proba(self.X_test)[:, 1]
                y_train_pred_proba = classifier.predict_proba(self.X_train)[:, 1]
                y_test_pred = classifier.predict(self.X_test)
                y_train_pred = classifier.predict(self.X_train)
                test_accuracy = accuracy_score(y_true=self.y_test, y_pred=y_test_pred)
                train_accuracy = accuracy_score(y_true=self.y_train, y_pred=y_train_pred)
                test_auc_score = roc_auc_score(y_true=self.y_test, y_score=y_test_pred_proba)
                train_auc_score = roc_auc_score(y_true=self.y_train, y_score=y_train_pred_proba)
                F1_score_binary = f1_score(y_true=self.y_test, y_pred=y_test_pred, average='binary')
                F1_score_micro = f1_score(y_true=self.y_test, y_pred=y_test_pred, average='micro')
                report = classification_report(self.y_test, y_test_pred, zero_division="warn")
                cm = confusion_matrix(self.y_test, y_test_pred)
                TP = cm[1][1]
                TN = cm[0][0]
                FP = cm[0][1]
                FN = cm[1][0]
                if TP + FP == 0:
                    precision = math.nan
                else:
                    precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                print(f"{self.classifier}_{self.model}_{i}")
                print(report)
                print(f'auc: {test_auc_score}')
                print(f'f1 binary: {F1_score_binary}, f1 micro: {F1_score_micro}')
                test_accuracy_array.append(test_accuracy)
                train_accuracy_array.append(train_accuracy)
                F1_score_micro_array.append(F1_score_micro)
                F1_score_binary_array.append(F1_score_binary)
                test_auc_array.append(test_auc_score)
                train_auc_array.append(train_auc_score)
                precision_array.append(precision)
                recall_array.append(recall)
                cm_list.append(cm)

            return (test_accuracy_array, train_accuracy_array, F1_score_binary_array, F1_score_micro_array,
                    test_auc_array, train_auc_array, precision_array, recall_array, cm_list)
