from sklearn.ensemble import HistGradientBoostingClassifier
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

pipeline_gnb = Pipeline([('GaussianNB', GaussianNB())])
pipeline_knn = Pipeline([('KNeighborsClassifier', KNeighborsClassifier())])
pipeline_rf = Pipeline([('RandomForestClassifier', RandomForestClassifier())])
pipeline_dt = Pipeline([('DecisionTreeClassifier', DecisionTreeClassifier())])
pipeline_xgb = Pipeline([('XGBClassifier', XGBClassifier(tree_method='gpu_hist'))])
pipeline_lgbm = Pipeline([('LGBMClassifier', LGBMClassifier(objective='binary'))])
pipeline_catB = Pipeline([('CatBoostClassifier', CatBoostClassifier(task_type="GPU", verbose=True))])
pipeline_hist = Pipeline([('HistGradientBoostingClassifier', HistGradientBoostingClassifier())])
pipeline_gradB = Pipeline([('GradientBoostingClassifier', GradientBoostingClassifier())])
pipeline_adaB = Pipeline([('AdaBoostClassifier', AdaBoostClassifier())])
pipeline_lr = Pipeline([('LogisticRegression', LogisticRegression())])
