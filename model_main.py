import pandas as pd
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from src.test_model import test_model, test_model_lists
from src.choose_device import get_default_device, to_device, DeviceDataLoader
from src.fit import f1
from DatasetsLoader import print_datasets_list, preprocess


def shuffle_in_unison(a, b):  # Shuffling the features and labels in unison.
    assert len(a) == len(
        b)  # In Python, the assert statement is used to continue the execute if the given condition evaluates to True.
    shuffled_a = np.empty(a.shape,
                          dtype=a.dtype)  # Return a new array of given shape and type, without initializing entries.
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def Oversampled_data(X_train_SMOTE, y_train_SMOTE, y_train, device):
    X_oversampled_0 = []
    X_oversampled_2 = []
    X_oversampled_3 = []
    for i in y_train_SMOTE[(y_train.shape[0]):]:
        if i == 0:
            X_oversampled_0.append(X_train_SMOTE[i])
        elif i == 2:
            X_oversampled_2.append(X_train_SMOTE[i])
        elif i == 3:
            X_oversampled_3.append(X_train_SMOTE[i])
    X_oversampled_0 = torch.from_numpy(np.array(X_oversampled_0))
    X_oversampled_0 = to_device(X_oversampled_0.float(), device)
    X_oversampled_2 = torch.from_numpy(np.array(X_oversampled_2))
    X_oversampled_2 = to_device(X_oversampled_2.float(), device)
    X_oversampled_3 = torch.from_numpy(np.array(X_oversampled_3))
    X_oversampled_3 = to_device(X_oversampled_3.float(), device)

    return X_oversampled_0, X_oversampled_2, X_oversampled_3


def GANs_two_class_real_data(X_train, y_train):  # Defining the real data for GANs
    X_real = []
    y_train = y_train.ravel()
    for i in range(len(y_train)):
        if int(y_train[i]) == 1:
            X_real.append(X_train[i])
    X_real = np.array(X_real)
    y_real = np.ones((X_real.shape[0],))
    return X_real, y_real


def model_eva(dataset_name):
    # load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess(dataset_name=dataset_name)

    # Calculating train and test accuracy and f1 score of non oversampled training data
    Normal_test_accuracy, Normal_train_accuracy, Normal_f1_score, Normal_auc_score, Normal_precision, Normal_recall, Normal_cm\
        = test_model_lists(X_train, y_train.ravel(), X_test, y_test.ravel(), 30, "Normal")

    print("Before OverSampling, counts of label '0': {}".format(sum(y_train == 0)))
    print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
    print("Before OverSampling, counts of label '2': {}".format(sum(y_train == 2)))
    print("Before OverSampling, counts of label '3': {}".format(sum(y_train == 3)))

    X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(X_train, y_train)

    print('After OverSampling, the shape of train_X: {}'.format(X_train_SMOTE.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train_SMOTE.shape))

    print("After OverSampling, counts of label '0': {}".format(sum(y_train_SMOTE == 0)))
    print("After OverSampling, counts of label '1': {}".format(sum(y_train_SMOTE == 1)))
    print("After OverSampling, counts of label '2': {}".format(sum(y_train_SMOTE == 2)))
    print("After OverSampling, counts of label '3': {}".format(sum(y_train_SMOTE == 3)))

    #### Calculating train and test accuracy and f1 score of SMOTE oversampled training data ####
    SMOTE_test_accuracy, SMOTE_train_accuracy, SMOTE_f1_score, SMOTE_auc_score,SMOTE_precision, SMOTE_recall, SMOTE_cm\
        = test_model_lists(X_train_SMOTE, y_train_SMOTE.ravel(), X_test, y_test.ravel(), 30, "SMOTE")

    device = get_default_device()
    # print(device)

    ##################### TWO CLASS ABALONE #####################
    ##### Oversampled data from SMOTE that is now to be passed in SMOTified GANs #####
    X_oversampled = X_train_SMOTE[(X_train.shape[0]):]
    X_oversampled = torch.from_numpy(X_oversampled)
    X_oversampled = to_device(X_oversampled.float(), device)

    # print(X_oversampled.shape)
    lr = 0.0002
    epochs = 150
    batch_size = 128

    X_real, y_real = GANs_two_class_real_data(X_train, y_train)  # Defining the real data to be put in GANs

    # Training our SMOTified GANs and GANs model and fetching their trained generators.
    generator_SG, generator_G = f1(X_train, y_train, X_train_SMOTE, y_train_SMOTE, X_real, y_real, X_oversampled,
                                   device, lr, epochs, batch_size, 1, 0)

    Trained_X_oversampled_SG = generator_SG(X_oversampled.float().to(device)).cpu().detach().numpy()
    Trained_SG_dataset = np.concatenate((X_train_SMOTE[:(X_train.shape[0])], Trained_X_oversampled_SG), axis=0)
    X_trained_SG, y_trained_SG = shuffle_in_unison(Trained_SG_dataset, y_train_SMOTE)

    #### Calculating train and test accuracy and f1 score of SMOTified GANs oversampled training data ####
    SG_test_accuracy, SG_train_accuracy, SG_f1_score, SG_auc_score, SG_precision, SG_recall, SG_cm\
        = test_model_lists(X_trained_SG, y_trained_SG.ravel(), X_test, y_test.ravel(), 30, "SMOTified GAN")

    GANs_noise = torch.randn((X_oversampled.shape[0]), (X_oversampled.shape[1]), device=device)
    Trained_X_oversampled_G = generator_G(GANs_noise.float().to(device)).cpu().detach().numpy()
    Trained_G_dataset = np.concatenate((X_train_SMOTE[:(X_train.shape[0])], Trained_X_oversampled_G), axis=0)
    X_trained_G, y_trained_G = shuffle_in_unison(Trained_G_dataset, y_train_SMOTE)

    #### Calculating train and test accuracy and f1 score of SMOTified GANs oversampled training data ####
    G_test_accuracy, G_train_accuracy, G_f1_score, G_auc_score, G_precision, G_recall, G_cm\
        = test_model_lists(X_trained_G, y_trained_G.ravel(), X_test, y_test.ravel(), 30, "GAN")

    print(Normal_test_accuracy)
    print(Normal_train_accuracy)
    print(Normal_f1_score)
    print(Normal_auc_score)
    print(SMOTE_test_accuracy)
    print(SMOTE_train_accuracy)
    print(SMOTE_f1_score)
    print(SMOTE_auc_score)
    print(SG_test_accuracy)
    print(SG_train_accuracy)
    print(SG_f1_score)
    print(SG_auc_score)
    print(G_test_accuracy)
    print(G_train_accuracy)
    print(G_f1_score)
    print(G_auc_score)

    data = {'Normal_test_accuracy': Normal_test_accuracy,
            'Normal_train_accuracy': Normal_train_accuracy,
            'Normal_f1_score': Normal_f1_score,
            'Normal_auc_score': Normal_auc_score,
            'Normal_precision': Normal_precision,
            'Normal_recall': Normal_recall,
            'Normal_cm': Normal_cm,
            'SMOTE_test_accuracy': SMOTE_test_accuracy,
            'SMOTE_train_accuracy': SMOTE_train_accuracy,
            'SMOTE_f1_score': SMOTE_f1_score,
            'SMOTE_auc_score': SMOTE_auc_score,
            'SMOTE_precision': SMOTE_precision,
            'SMOTE_recall': SMOTE_recall,
            'SMOTE_cm': SMOTE_cm,
            'SG_test_accuracy': SG_test_accuracy,
            'SG_train_accuracy': SG_train_accuracy,
            'SG_f1_score': SG_f1_score,
            'SG_auc_score': SG_auc_score,
            'SG_precision': SG_precision,
            'SG_recall': SG_recall,
            'SG_cm': SG_cm,
            'G_test_accuracy': G_test_accuracy,
            'G_train_accuracy': G_train_accuracy,
            'G_f1_score': G_f1_score,
            'G_auc_score': G_auc_score,
            'G_precision': G_precision,
            'G_recall': G_recall,
            'G_cm': G_cm
            }
    df = pd.DataFrame(data)
    df.to_csv('./results/africa.csv', index=False)

    import math
    def average(lst):
        cleaned_list = [x for x in lst if x is not math.nan]

        if len(cleaned_list) == 0:
            return None

        return sum(cleaned_list) / len(cleaned_list)

    Normal_test_accuracy_avg = average(Normal_test_accuracy)
    Normal_train_accuracy_avg = average(Normal_train_accuracy)
    Normal_f1_score_avg = average(Normal_f1_score)
    Normal_auc_score_avg = average(Normal_auc_score)
    Normal_precision_avg = average(Normal_precision)
    Normal_recall_avg = average(Normal_recall)
    SMOTE_test_accuracy_avg = average(SMOTE_test_accuracy)
    SMOTE_train_accuracy_avg = average(SMOTE_train_accuracy)
    SMOTE_f1_score_avg = average(SMOTE_f1_score)
    SMOTE_auc_score_avg = average(SMOTE_auc_score)
    SMOTE_precision_avg = average(SMOTE_precision)
    SMOTE_recall_avg = average(SMOTE_recall)
    SG_test_accuracy_avg = average(SG_test_accuracy)
    SG_train_accuracy_avg = average(SG_train_accuracy)
    SG_f1_score_avg = average(SG_f1_score)
    SG_auc_score_avg = average(SG_auc_score)
    SG_precision_avg = average(SG_precision)
    SG_recall_avg = average(SG_recall)
    G_test_accuracy_avg = average(G_test_accuracy)
    G_train_accuracy_avg = average(G_train_accuracy)
    G_f1_score_avg = average(G_f1_score)
    G_auc_score_avg = average(G_auc_score)
    G_precision_avg = average(G_precision)
    G_recall_avg = average(G_recall)

    data = {'Normal_test_accuracy': [Normal_test_accuracy_avg],
            'Normal_train_accuracy': [Normal_train_accuracy_avg],
            'Normal_f1_score': [Normal_f1_score_avg],
            'Normal_auc_score': [Normal_auc_score_avg],
            'Normal_precision': [Normal_precision_avg],
            'Normal_recall': [Normal_recall_avg],
            'SMOTE_test_accuracy': [SMOTE_test_accuracy_avg],
            'SMOTE_train_accuracy': [SMOTE_train_accuracy_avg],
            'SMOTE_f1_score': [SMOTE_f1_score_avg],
            'SMOTE_auc_score': [SMOTE_auc_score_avg],
            'SMOTE_precision': [SMOTE_precision_avg],
            'SMOTE_recall': [SMOTE_recall_avg],
            'SG_test_accuracy': [SG_test_accuracy_avg],
            'SG_train_accuracy': [SG_train_accuracy_avg],
            'SG_f1_score': [SG_f1_score_avg],
            'SG_auc_score': [SG_auc_score_avg],
            'SG_precision': [SG_precision_avg],
            'SG_recall': [SG_recall_avg],
            'G_test_accuracy': [G_test_accuracy_avg],
            'G_train_accuracy': [G_train_accuracy_avg],
            'G_f1_score': [G_f1_score_avg],
            'G_auc_score': [G_auc_score_avg],
            'G_precision': [G_precision_avg],
            'G_recall': [G_recall_avg]}

    df = pd.DataFrame(data)
    df.to_csv('./results/africa_avg.csv', index=False)
    print(Normal_precision)
    #print not number value in Normal_precision
    print(type(Normal_precision_avg))



