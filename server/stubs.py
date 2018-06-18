from server.base import *


class StubActivityContentClassifier(ActivityContentClassifier):
    def classify(self, content):
        return [{'id': 0, 'name': '旅遊'}, {'id': 1, 'name': '運動'}]


class StubUserPreferencesPredictor(UserPreferencesPredictor):
    def get_possibility(self, user,
                        user_association_histories,
                        target_activity):
        return 0.5
