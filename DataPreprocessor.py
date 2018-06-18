import json
from datetime import datetime

import numpy as np
import collections
import dateutil.parser
from activity_sources import get_source_name_one_hot
from keras.preprocessing.sequence import pad_sequences
from words_preprocessing_utils import *

ACTIVITY_TAGS = ['福利', '獎金', '文藝', '運動', '旅遊', '學習', '娛樂', '美食', '公告']
CITIES = ['臺北市', '新北市', '臺中市', '臺南市', '高雄市', '桃園市', '基隆市', '新竹市', '嘉義市', '宜蘭市',
          '花蓮市', '彰化市', '屏東市', '臺東市', '苗栗市', '南投市', '斗六市', '馬公市', '斗六市',
          '竹北市', '太保市', '朴子市', '員林市', '頭份市']

MAX_ASSOCIATION_SEQUENCES = 200
WORD_VECTOR_NORMALIZED_FACTOR = 100


class DataPreprocessor:
    def __init__(self):
        self.activities = None
        self.word_list = None
        self.word_to_index = None
        self.prepare_activities()
        self.parse_date = lambda s: DataPreprocessor.parse_date_string(s)

    @staticmethod
    def parse_date_string(s):
        assert isinstance(s, str)
        date = dateutil.parser.parse(s)
        return date

    @staticmethod
    def create_one_hot_tags_label(activity_tags):
        one_hot = np.zeros(len(ACTIVITY_TAGS))
        for tag in activity_tags:
            index = ACTIVITY_TAGS.index(tag['name'])
            one_hot[index] = 1
        return one_hot

    @staticmethod
    def create_one_hot_city_label(city_name):
        one_hot = np.zeros(len(CITIES))
        one_hot[CITIES.index(city_name)] = 1
        return one_hot

    def prepare_activities(self):
        print('Activities preparing...')
        with open('activities.json', 'r', encoding='utf-8') as fr:
            self.activities = json.load(fr)['data']
        print('Activities prepared...')

    def get_tagged_activities(self):
        return [a for a in self.activities
                if len(a['ActivityTags']) != 0 and
                len([tag for tag in a['ActivityTags'] if tag['name'] == '垃圾']) == 0]  # exclude 垃圾

    def get_words_frequency_preprocessing(self):
        activity_texts = []
        labels = []

        for activity in self.get_tagged_activities():
            activity_texts.append(clean_html(activity['title'] + activity['content']))
            labels.append(DataPreprocessor.create_one_hot_tags_label(activity['ActivityTags']))

        data, self.word_list, self.word_to_index = get_word_frequency_vectors(activity_texts,
                                                                              output_file_name='word_list.txt')
        return np.array(data), np.array(labels), self.word_list

    def get_words_embedding_preprocessing(self):
        labels = []
        activity_texts = []

        for activity in self.get_tagged_activities():
            activity_texts.append(clean_html(activity['title'] + activity['content']))
            labels.append(DataPreprocessor.create_one_hot_tags_label(activity['ActivityTags']))

        data = get_word_embedding_vectors(activity_texts)
        return np.array(data, dtype=np.float64) / WORD_VECTOR_NORMALIZED_FACTOR, np.array(labels)

    @staticmethod
    def enumerate_sequences_labels(data):
        sequences = []
        labels = []
        for sequence in data:
            selections = [selection for selection in sequence
                          if np.any(selection)]  # filter off the selections with all zeros
            for i in range(len(selections) - 1):
                enumerated_seqs = [seq for seq in selections[:i + 1]]
                label = selections[i + 1]
                sequences.append(enumerated_seqs)
                labels.append(label)

            if len(sequences) % 50 == 0:
                print(len(sequences), ' sequences have been enumerated.')

        return np.array(sequences), np.array(labels)

    @staticmethod
    def get_activity_feature(activity):
        content = clean_html(activity['title'] + activity['content'])
        word_vector = get_word_embedding_vector(content).tolist()
        one_hot_tags = DataPreprocessor.create_one_hot_tags_label(activity['ActivityTags']).tolist()
        one_hot_source = get_source_name_one_hot(activity['source']).tolist()
        return word_vector + one_hot_tags + one_hot_source

    @staticmethod
    def get_user_feature(user):
        one_hot_city = DataPreprocessor.create_one_hot_city_label(user['City']['name']).tolist()
        return [user['age'], int(user['gender'])] + one_hot_city

    @staticmethod
    def get_association_feature(association):
        activity_feature = DataPreprocessor.get_activity_feature(association['activity'])
        user_feature = DataPreprocessor.get_user_feature(association['user'])
        return activity_feature + user_feature + [association['action']]

    @staticmethod
    def get_user_associations_sequence(associations):
        return [DataPreprocessor.get_association_feature(association)
                        for association in associations]

    def get_association_training_sequences_labels(self):
        with open('associations.json', 'r', encoding='utf-8') as fr:
            associations = json.load(fr)['data']
            print('Associations got ', len(associations))

        # list associations for each user
        userid_to_associations = collections.defaultdict(list)
        for association in associations:
            userid_to_associations[association['user']['id']].append(association)

        # sort each user's association by there date
        for userid, associations in userid_to_associations.items():
            userid_to_associations[userid] = sorted(associations, key=lambda a: self.parse_date(a['date']))

        # create all sequences of association features of each user
        data = [DataPreprocessor.get_user_associations_sequence(associations)
                for associations in userid_to_associations.values()]

        return DataPreprocessor.enumerate_sequences_labels(data)

    def get_cnn_representation_training_data_labels(self):
        activity_texts = []
        labels = []

        for activity in self.get_tagged_activities():
            activity_texts.append(clean_html(activity['title'] + activity['content']))
            labels.append(DataPreprocessor.create_one_hot_tags_label(activity['ActivityTags']))

        data, self.word_list, self.word_to_index = get_word_frequency_vectors(activity_texts,
                                                                              output_file_name='word_list.txt')
        return np.array(data), np.array(labels), self.word_list

if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    sequences, labels = preprocessor.get_association_training_sequences_labels()
    assert len(sequences) == len(labels)
    print(sequences, labels)
