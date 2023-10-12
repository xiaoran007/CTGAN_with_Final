import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from statistics import stdev
import math

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class MLP:
    def __init__(self, X_train, y_train, X_test, y_test, test_name, test_round, batch_size=8192, epochs=30, model=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.test_name = test_name
        self.test_round = test_round
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = model

    def Set_High_Normal(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(1)
        ])

        self.model = model

    def Set_Low_Normal(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(1)
        ])

        self.model = model

    def Set_High_UnNormal(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model = model

    def Set_Low_UnNormal(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model = model

    def Test(self):
        self.model.compile(optimizer='adam',
                           loss='mean_absolute_error',
                           metrics=['accuracy', tf.keras.metrics.AUC()])

        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, verbose=1, batch_size=self.batch_size)
        test_loss, test_accuracy, test_auc = self.model.evaluate(self.X_test, self.y_test, verbose=1, batch_size=self.batch_size)
        train_loss, train_accuracy, train_auc = self.model.evaluate(self.X_train, self.y_train, verbose=1, batch_size=self.batch_size)
        y_predicts = self.model.predict(self.X_test, verbose=1, batch_size=self.batch_size)
        y_predicts = np.ravel((y_predicts > 0.5) * 1)

        F1_score_binary = f1_score(self.y_test, y_predicts, average='binary')
        F1_score_micro = f1_score(self.y_test, y_predicts, average='micro')
        cm = confusion_matrix(self.y_test, y_predicts)
        TP = cm[1][1]
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        if TP + FP == 0:
            precision = math.nan
        else:
            precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        print("----------------------------------------")
        print(f"{self.test_name} - {self.test_round}")
        print(classification_report(self.y_test, y_predicts, zero_division=0))
        print('Test accuracy:', test_accuracy)
        print('AUC :', test_auc)
        print('F1 binary:', F1_score_binary)
        print('F1 micro:', F1_score_micro)

        return test_accuracy, train_accuracy, F1_score_binary, F1_score_micro, test_auc, train_auc, precision, recall, cm


def MLPClassifier(X_train, y_train, X_test, y_test, test_name, rounds=30, mlp_type='Low_UnNormal'):
    test_accuracy_array = []
    train_accuracy_array = []
    F1_score_binary_array = []
    F1_score_micro_array = []
    test_auc_array = []
    train_auc_array = []
    precision_array = []
    recall_array = []
    cm_list = []

    for i in range(rounds):
        mlp_obj = MLP(X_train, y_train, X_test, y_test, test_name=test_name, test_round=i)
        if mlp_type == 'High_Normal':
            mlp_obj.Set_High_Normal()
        elif mlp_type == 'Low_Normal':
            mlp_obj.Set_Low_Normal()
        elif mlp_type == 'High_UnNormal':
            mlp_obj.Set_High_UnNormal()
        elif mlp_type == 'Low_UnNormal':
            mlp_obj.Set_Low_UnNormal()
        test_accuracy, train_accuracy, F1_score_binary, F1_score_micro, test_auc, train_auc, precision, recall, cm = mlp_obj.Test()
        test_accuracy_array.append(test_accuracy)
        train_accuracy_array.append(train_accuracy)
        F1_score_micro_array.append(F1_score_micro)
        F1_score_binary_array.append(F1_score_binary)
        test_auc_array.append(test_auc)
        train_auc_array.append(train_auc)
        precision_array.append(precision)
        recall_array.append(recall)
        cm_list.append(cm)

    return (test_accuracy_array, train_accuracy_array, F1_score_binary_array, F1_score_micro_array, test_auc_array,
            train_auc_array, precision_array, recall_array, cm_list)

