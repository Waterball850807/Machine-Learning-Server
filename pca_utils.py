from sklearn.decomposition import PCA
import numpy as np
import pickle

_default_file_name = 'pca.pickle'


def save_fit_PCA_and_transform(data, file_name=_default_file_name):
    pca = PCA(n_components='mle', svd_solver='full', copy=True)
    pca_data = np.array(pca.fit_transform(data, y=None))
    with open(file_name, 'wb') as f:
        pickle.dump(pca, f)
    return pca_data


def load_PCA_model(file_name=_default_file_name):
    with open(_default_file_name, 'rb') as f:
        return pickle.load(f)