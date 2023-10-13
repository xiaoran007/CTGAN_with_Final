from imblearn.over_sampling import SMOTE
from src.choose_device import get_default_device, to_device
from src.fit import f1
from ctgan import CTGAN
import numpy as np
import torch


class DataGenerator:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, model='None'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.generatorG = None
        self.generatorSG = None
        self.device = get_default_device()
        self.X_train_SMOTE = None
        self.y_train_SMOTE = None
        self.CTGAN = None

    def GenerateBaselineData(self):
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test

    def GenerateSMOTEData(self):
        X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(self.X_train, self.y_train)
        return X_train_SMOTE, y_train_SMOTE, self.X_val, self.y_val, self.X_test, self.y_test

    def GenerateGANData(self):
        if self.generatorG is None or self.X_train_SMOTE is None:
            print("Err.")
        else:
            X_oversampled = self.X_train_SMOTE[(self.X_train.shape[0]):]
            X_oversampled = torch.from_numpy(X_oversampled)
            X_oversampled = to_device(X_oversampled.float(), self.device)
            GANs_noise = torch.randn((X_oversampled.shape[0]), (X_oversampled.shape[1]), device=self.device)
            Trained_X_oversampled_G = self.generatorG(GANs_noise.float().to(self.device)).cpu().detach().numpy()
            Trained_G_dataset = np.concatenate((self.X_train_SMOTE[:(self.X_train.shape[0])], Trained_X_oversampled_G), axis=0)
            X_trained_G, y_trained_G = shuffle_in_unison(Trained_G_dataset, self.y_train_SMOTE)
            return X_trained_G, y_trained_G, self.X_val, self.y_val, self.X_test, self.y_test

    def GenerateSGANData(self):
        if self.generatorSG is None or self.X_train_SMOTE is None:
            print("Err.")
        else:
            X_oversampled = self.X_train_SMOTE[(self.X_train.shape[0]):]
            X_oversampled = torch.from_numpy(X_oversampled)
            X_oversampled = to_device(X_oversampled.float(), self.device)
            Trained_X_oversampled_SG = self.generatorSG(X_oversampled.float().to(self.device)).cpu().detach().numpy()
            Trained_SG_dataset = np.concatenate((self.X_train_SMOTE[:(self.X_train.shape[0])], Trained_X_oversampled_SG), axis=0)
            X_trained_SG, y_trained_SG = shuffle_in_unison(Trained_SG_dataset, self.y_train_SMOTE)
            return X_trained_SG, y_trained_SG, self.X_val, self.y_val, self.X_test, self.y_test

    def GenerateCTGANData(self):
        if self.CTGAN is None:
            print("Err.")
        else:
            num_samples = np.count_nonzero(self.y_train == 0) - np.count_nonzero(self.y_train == 1)
            synthetic_data_x = self.CTGAN.sample(num_samples)
            synthetic_data_y = np.ones(num_samples)
            X_train_ctgan = np.concatenate([synthetic_data_x, self.X_train], axis=0)
            y_train_ctgan = np.concatenate([synthetic_data_y, self.y_train], axis=0)
            return X_train_ctgan, y_train_ctgan, self.X_val, self.y_val, self.X_test, self.y_test

    def Set_GAN(self):
        device = self.device
        if self.X_train_SMOTE is None:
            self.Set_SMOTE()
        X_oversampled = self.X_train_SMOTE[(self.X_train.shape[0]):]
        X_oversampled = torch.from_numpy(X_oversampled)
        X_oversampled = to_device(X_oversampled.float(), device)

        lr = 0.0002
        epochs = 150
        batch_size = 128

        X_real, y_real = GANs_two_class_real_data(self.X_train, self.y_train)

        generator_SG, generator_G = f1(self.X_train, self.y_train, self.X_train_SMOTE, self.y_train_SMOTE, X_real, y_real, X_oversampled,
                                       device, lr, epochs, batch_size, 1, 0)

        self.generatorSG = generator_SG
        self.generatorG = generator_G

    def Set_SMOTE(self):
        X_train_SMOTE, y_train_SMOTE, _, _, _, _ = self.GenerateSMOTEData()
        self.X_train_SMOTE = X_train_SMOTE
        self.y_train_SMOTE = y_train_SMOTE

    def Set_CTGAN(self):
        self.CTGAN = CTGAN(batch_size=500, epochs=100, cuda=True, verbose=False)
        X_train_fraud = self.X_train[self.y_train == 1]
        self.CTGAN.fit(X_train_fraud)


def GANs_two_class_real_data(X_train, y_train):  # Defining the real data for GANs
    X_real = []
    y_train = y_train.ravel()
    for i in range(len(y_train)):
        if int(y_train[i]) == 1:
            X_real.append(X_train[i])
    X_real = np.array(X_real)
    y_real = np.ones((X_real.shape[0],))
    return X_real, y_real


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


if __name__ == '__main__':
    pass
