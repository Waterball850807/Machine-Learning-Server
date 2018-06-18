import numpy as np
from keras.models import load_model
from base import *
from keras.preprocessing.sequence import pad_sequences
from words_preprocessing_utils import get_word_embedding_vector
from DataPreprocessor import DataPreprocessor, MAX_ASSOCIATION_SEQUENCES, WORD_VECTOR_NORMALIZED_FACTOR, ACTIVITY_TAGS


def count_cosine_distance(narray1, narray2):
    return np.dot(narray1, narray2) / ( (np.dot(narray1, narray1) **.5) * (np.dot(narray2, narray2) ** .5) )


class KerasActivityContentClassifier(ActivityContentClassifier):

    def __init__(self, model_file_path):
        print("Loading classifier's model.")
        self.model = load_model(model_file_path)
        print("Classifier's model summary:")
        self.model.summary()

    def classify(self, content):
        feature = get_word_embedding_vector(content) / WORD_VECTOR_NORMALIZED_FACTOR
        x = np.reshape(feature, (1, feature.shape[0]))
        y = self.model.predict(x)[0]
        sorted_indices = [i for i in np.argsort(y)[::-1][:3] if y[i] >= 0.5]  # get top 3 classifications
        if len(sorted_indices) == 0:
            return ['垃圾']
        else:
            tag_names = []
            for index in sorted_indices:
                tag_names.append(ACTIVITY_TAGS[index])


class KerasUserPreferencesPredictor(UserPreferencesPredictor):

    def __init__(self, model_file_path):
        print("Loading PreferencesPredictor's model.")
        self.model = load_model(model_file_path)
        print("PreferencesPredictor's model summary:")
        self.model.summary()

    def get_possibility(self, user,
                            user_association_histories,
                            target_activity):
        activity_feature = DataPreprocessor.get_activity_feature(target_activity)
        sequences = DataPreprocessor.get_user_associations_sequence(user_association_histories)
        sequences = pad_sequences(sequences, maxlen=MAX_ASSOCIATION_SEQUENCES)

        x = np.reshape(sequences, (1, sequences.shape[0], sequences.shape[1]))
        y = self.model.predict(x)[0]

        possibility = count_cosine_distance(y, activity_feature)
        return possibility
