import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from statistics import stdev
import math

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class test_model():        #parent class
    def __init__(self, X_train, y_train, X_test, y_test, name, epochs):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.name = name
        self.epochs = epochs

    def __call__(self):
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam',
                      loss='mean_absolute_error',
                      metrics=['accuracy', tf.keras.metrics.AUC()])

        model.fit(self.X_train, self.y_train, epochs=30, verbose=1, batch_size=8192)
        test_loss, test_accuracy, test_auc = model.evaluate(self.X_test, self.y_test, verbose=1, batch_size=8192)
        train_loss, train_accuracy, train_auc = model.evaluate(self.X_train, self.y_train, verbose=1, batch_size=8192)
        y_preds = model.predict(self.X_test, verbose=1, batch_size=8192)
        y_preds = np.ravel((y_preds > 0.5) * 1)

        F1_score = f1_score(self.y_test, y_preds, average='binary')
        cm = confusion_matrix(self.y_test, y_preds)
        TP = cm[1][1]
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        if TP+FP == 0:
            precision = math.nan
        else:
            precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
        print("----------------------------------------")
        print(f"{self.name} - {self.epochs}")
        print(classification_report(self.y_test, y_preds, zero_division=0))
        print('Test accuracy:', test_accuracy)
        print('AUC :', test_auc)
        # disp.plot()
        # plt.title(f"{self.name} - {self.epochs}")
        # plt.show()

        return test_accuracy, train_accuracy, F1_score, test_auc, train_auc, precision, recall, cm

def test_model_lists(X_train, y_train, X_test, y_test, no_of_trainings, name):
    test_accuracy_array = []
    train_accuracy_array = []
    f1_score_array = []
    test_auc_array = []
    precision_array = []
    recall_array = []
    cm_list = []

    for i in range(no_of_trainings):
        test_model_object = test_model(X_train, y_train.ravel(), X_test, y_test.ravel(), name, i)
        test_accuracy, train_accuracy, F1_score, test_auc, train_auc, precision, recall,cm  = test_model_object()
        test_accuracy_array.append(test_accuracy)
        train_accuracy_array.append(train_accuracy)
        f1_score_array.append(F1_score)
        test_auc_array.append(test_auc)
        precision_array.append(precision)
        recall_array.append(recall)
        cm_list.append(cm)

    return test_accuracy_array, train_accuracy_array, f1_score_array, test_auc_array, precision_array, recall_array, cm_list



