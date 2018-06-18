from base import *


class StubActivityContentClassifier(ActivityContentClassifier):
    def classify(self, content):
        return ['旅遊', '運動']


class StubUserPreferencesPredictor(UserPreferencesPredictor):
    def get_possibility(self, user,
                        user_association_histories,
                        target_activity):
        return 0.5
