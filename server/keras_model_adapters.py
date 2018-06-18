from keras.models import load_model

from server.base import *


class KerasActivityContentClassifier(ActivityContentClassifier):

    def __init__(self, model_file_path):
        print("Loading classifier's model.")
        self.model = load_model(model_file_path)
        print("Classifier's model summary:")
        self.model.summary()

    def classify(self, content):
        pass


class KerasUserPreferencesPredictor(UserPreferencesPredictor):

    def __init__(self, model_file_path):
        print("Loading PreferencesPredictor's model.")
        self.model = load_model(model_file_path)
        print("PreferencesPredictor's model summary:")
        self.model.summary()

    def get_possibility(self, user,
                            user_association_histories,
                            target_activity):
        pass