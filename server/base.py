

class ActivityContentClassifier:

    def classify(self, content):
        """
        :param content: pure text of the content of the activity
        :return: list of possible activity tags
        """
        pass


class UserPreferencesPredictor:

    def get_possibility(self, user,
                            user_association_histories,
                            target_activity):
        """
        :param user: obj of the user
        :param user_association_histories: the histories of associations list the user already had
        :param target_activity: obj of the activity that needs to be predicted if the user prefers
        :return: possibility of the user preferring the activity
        """
        pass