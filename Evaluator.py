from sklearn.model_selection import train_test_split
import DatasetsLoader
import DataGenerator
import Classifier
import pandas as pd


def split_dataset_random(X, y):
    # Split dataset into 7:1:2 for training : validation : testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=None, stratify=y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


class Evaluator:
    def __init__(self, dataset_name, classifier_name, rounds=30):
        self.dataset_name = dataset_name
        self.classifier_name = classifier_name
        self.rounds = rounds

    def evaluate(self):
        dataset = DatasetsLoader.Dataset(dataset_name=self.dataset_name)
        X, y = dataset.GetDataset()
        dataframe = self.CreatDataframe()
        for i in range(self.rounds):
            # Normal data
            X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_random(X, y)

            # SMOTE data
            SMOTE_gen_obj = DataGenerator.DataGenerator(X_train, y_train, X_val, y_val, X_test, y_test, model='SMOTE')
            SMOTE_gen_obj.Set_SMOTE()
            X_train_SMOTE, y_train_SMOTE, X_val_SMOTE, y_val_SMOTE, X_test_SMOTE, y_test_SMOTE = SMOTE_gen_obj.GenerateSMOTEData()

            # GAN and SMOTE-GAN data
            GAN_SGAN_gen_obj = DataGenerator.DataGenerator(X_train, y_train, X_val, y_val, X_test, y_test, model='GAN')
            GAN_SGAN_gen_obj.Set_SMOTE()
            GAN_SGAN_gen_obj.Set_GAN()
            X_train_GAN, y_train_GAN, X_val_GAN, y_val_GAN, X_test_GAN, y_test_GAN = GAN_SGAN_gen_obj.GenerateGANData()
            X_train_SGAN, y_train_SGAN, X_val_SGAN, y_val_SGAN, X_test_SGAN, y_test_SGAN = GAN_SGAN_gen_obj.GenerateSGANData()

            # CTGAN data
            CTGAN_gen_obj = DataGenerator.DataGenerator(X_train, y_train, X_val, y_val, X_test, y_test, model='CTGAN')
            CTGAN_gen_obj.Set_CTGAN()
            X_train_CTGAN, y_train_CTGAN, X_val_CTGAN, y_val_CTGAN, X_test_CTGAN, y_test_CTGAN = CTGAN_gen_obj.GenerateCTGANData()

            # Normal result
            Normal_eva_obj = Classifier.Classifier(X_train, y_train, X_val, y_val, X_test, y_test,
                                                   classifier=self.classifier_name, model='Normal',
                                                   in_round=i, epoch=30, batch_size=8192)
            Normal_test_accuracy, Normal_train_accuracy, Normal_F1_score_binary, Normal_F1_score_micro, Normal_test_auc, \
                Normal_train_auc, Normal_precision, Normal_recall, Normal_cm = Normal_eva_obj.TestSingle()

            # SMOTE result
            SMOTE_eva_obj = Classifier.Classifier(X_train_SMOTE, y_train_SMOTE, X_val_SMOTE, y_val_SMOTE, X_test_SMOTE,
                                                  y_test_SMOTE, classifier=self.classifier_name, model='SMOTE',
                                                  in_round=i, epoch=30, batch_size=8192)
            SMOTE_test_accuracy, SMOTE_train_accuracy, SMOTE_F1_score_binary, SMOTE_F1_score_micro, SMOTE_test_auc_score,\
                SMOTE_train_auc_score, SMOTE_precision, SMOTE_recall, SMOTE_cm = SMOTE_eva_obj.TestSingle()

            # GAN result
            GAN_eva_obj = Classifier.Classifier(X_train_GAN, y_train_GAN, X_val_GAN, y_val_GAN, X_test_GAN, y_test_GAN,
                                                classifier=self.classifier_name, model='GAN',
                                                in_round=i, epoch=30, batch_size=8192)
            GAN_test_accuracy, GAN_train_accuracy, GAN_F1_score_binary, GAN_F1_score_micro, GAN_test_auc_score, \
                GAN_train_auc_score, GAN_precision, GAN_recall, GAN_cm = GAN_eva_obj.TestSingle()

            # SGAN result
            SGAN_eva_obj = Classifier.Classifier(X_train_SGAN, y_train_SGAN, X_val_SGAN, y_val_SGAN, X_test_SGAN, y_test_SGAN,
                                                 classifier=self.classifier_name, model='SMOTE-GAN',
                                                 in_round=i, epoch=30, batch_size=8192)
            SGAN_test_accuracy, SGAN_train_accuracy, SGAN_F1_score_binary, SGAN_F1_score_micro, SGAN_test_auc, SGAN_train_auc, \
                SGAN_precision, SGAN_recall, SGAN_cm = SGAN_eva_obj.TestSingle()

            # CTGAN result
            CTGAN_eva_obj = Classifier.Classifier(X_train_CTGAN, y_train_CTGAN, X_val_CTGAN, y_val_CTGAN, X_test_CTGAN, y_test_CTGAN,
                                                  classifier=self.classifier_name, model='SMOTE-GAN',
                                                  in_round=i, epoch=30, batch_size=8192)
            CTGAN_test_accuracy, CTGAN_train_accuracy, CTGAN_F1_score_binary, CTGAN_F1_score_micro, CTGAN_test_auc, CTGAN_train_auc, \
                CTGAN_precision, CTGAN_recall, CTGAN_cm = CTGAN_eva_obj.TestSingle()

            data = {'Normal_test_accuracy': [Normal_test_accuracy],
                    'Normal_train_accuracy': [Normal_train_accuracy],
                    'Normal_f1_score_binary': [Normal_F1_score_binary],
                    'Normal_f1_score_micro': [Normal_F1_score_micro],
                    'Normal_test_auc_score': [Normal_test_auc],
                    'Normal_train_auc_score': [Normal_train_auc],
                    'Normal_precision': [Normal_precision],
                    'Normal_recall': [Normal_recall],
                    'Normal_cm': [Normal_cm],
                    'SMOTE_test_accuracy': [SMOTE_test_accuracy],
                    'SMOTE_train_accuracy': [SMOTE_train_accuracy],
                    'SMOTE_f1_score_binary': [SMOTE_F1_score_binary],
                    'SMOTE_f1_score_micro': [SMOTE_F1_score_micro],
                    'SMOTE_test_auc_score': [SMOTE_test_auc_score],
                    'SMOTE_train_auc_score': [SMOTE_train_auc_score],
                    'SMOTE_precision': [SMOTE_precision],
                    'SMOTE_recall': [SMOTE_recall],
                    'SMOTE_cm': [SMOTE_cm],
                    'GAN_test_accuracy': [GAN_test_accuracy],
                    'GAN_train_accuracy': [GAN_train_accuracy],
                    'GAN_f1_score_binary': [GAN_F1_score_binary],
                    'GAN_f1_score_micro': [GAN_F1_score_micro],
                    'GAN_test_auc_score': [GAN_test_auc_score],
                    'GAN_train_auc_score': [GAN_train_auc_score],
                    'GAN_precision': [GAN_precision],
                    'GAN_recall': [GAN_recall],
                    'GAN_cm': [GAN_cm],
                    'SGAN_test_accuracy': [SGAN_test_accuracy],
                    'SGAN_train_accuracy': [SGAN_train_accuracy],
                    'SGAN_f1_score_binary': [SGAN_F1_score_binary],
                    'SGAN_f1_score_micro': [SGAN_F1_score_micro],
                    'SGAN_test_auc_score': [SGAN_test_auc],
                    'SGAN_train_auc_score': [SGAN_train_auc],
                    'SGAN_precision': [SGAN_precision],
                    'SGAN_recall': [SGAN_recall],
                    'SGAN_cm': [SGAN_cm],
                    'CTGAN_test_accuracy': [CTGAN_test_accuracy],
                    'CTGAN_train_accuracy': [CTGAN_train_accuracy],
                    'CTGAN_f1_score_binary': [CTGAN_F1_score_binary],
                    'CTGAN_f1_score_micro': [CTGAN_F1_score_micro],
                    'CTGAN_test_auc_score': [CTGAN_test_auc],
                    'CTGAN_train_auc_score': [CTGAN_train_auc],
                    'CTGAN_precision': [CTGAN_precision],
                    'CTGAN_recall': [CTGAN_recall],
                    'CTGAN_cm': [CTGAN_cm],
                    }
            result = pd.DataFrame(data)
            dataframe = pd.concat([dataframe, result], ignore_index=True)

        return dataframe

    @staticmethod
    def CreatDataframe():
        data = {'Normal_test_accuracy': [],
                'Normal_train_accuracy': [],
                'Normal_f1_score_binary': [],
                'Normal_f1_score_micro': [],
                'Normal_test_auc_score': [],
                'Normal_train_auc_score': [],
                'Normal_precision': [],
                'Normal_recall': [],
                'Normal_cm': [],
                'SMOTE_test_accuracy': [],
                'SMOTE_train_accuracy': [],
                'SMOTE_f1_score_binary': [],
                'SMOTE_f1_score_micro': [],
                'SMOTE_test_auc_score': [],
                'SMOTE_train_auc_score': [],
                'SMOTE_precision': [],
                'SMOTE_recall': [],
                'SMOTE_cm': [],
                'GAN_test_accuracy': [],
                'GAN_train_accuracy': [],
                'GAN_f1_score_binary': [],
                'GAN_f1_score_micro': [],
                'GAN_test_auc_score': [],
                'GAN_train_auc_score': [],
                'GAN_precision': [],
                'GAN_recall': [],
                'GAN_cm': [],
                'SGAN_test_accuracy': [],
                'SGAN_train_accuracy': [],
                'SGAN_f1_score_binary': [],
                'SGAN_f1_score_micro': [],
                'SGAN_test_auc_score': [],
                'SGAN_train_auc_score': [],
                'SGAN_precision': [],
                'SGAN_recall': [],
                'SGAN_cm': [],
                'CTGAN_test_accuracy': [],
                'CTGAN_train_accuracy': [],
                'CTGAN_f1_score_binary': [],
                'CTGAN_f1_score_micro': [],
                'CTGAN_test_auc_score': [],
                'CTGAN_train_auc_score': [],
                'CTGAN_precision': [],
                'CTGAN_recall': [],
                'CTGAN_cm': [],
                }
        df = pd.DataFrame(data)
        return df






