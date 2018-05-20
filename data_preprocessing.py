import json
import operator
import numpy as np
from sklearn import cluster, datasets
import jieba_utils
import arff_making
import jieba
import collections

puncts = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')

jieba.set_dictionary('dict.txt.big')


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


class DataPreprocessor:
    OUTPUT_WORDS_FILE_PATH = "wordset.txt"
    ACTIVITY_TAGS = ['靜態', '動態', '室外', '室內', '福利', '酬勞', '垃圾', '文藝', '運動', '旅遊', '學習', '娛樂', '公告']

    def __init__(self, word_index_dict=None, output_words=False):
        self.activities = None
        self.output_words = output_words
        self.word_count_dict = collections.defaultdict(int)
        self.word_index_dict = word_index_dict

    def start_preprocessing(self, filename='activities.json'):
        self.activities = sorted(load_json(filename)['data'], key=lambda a: a['id'])
        if not self.word_index_dict:
            print("The word index dict is not specified, start the word collecting process.")
            self.word_index_dict = dict()
            self.feed_word_count_dict(self.activities)
            self.feed_word_index_dict()

    def feed_word_count_dict(self, activities):
        for activity in activities:
            words = self.cut_activity_to_words(activity)
            for word in words:
                self.word_count_dict[word] += 1
        print("The words have been counted in a dict.")

    @staticmethod
    def cut_activity_to_words(activity):
        text = activity['title'].strip() * 3 + activity['content'].strip()
        return jieba_utils.cut(text)

    def feed_word_index_dict(self):
        sorted_word_counts = [(k, v) for k, v in (sorted(self.word_count_dict.items(), key=operator.itemgetter(1)))]
        all_words = [word for word, count in sorted_word_counts]

        if self.output_words:
            with open(DataPreprocessor.OUTPUT_WORDS_FILE_PATH, 'w+', encoding='utf-8') as f:
                f.writelines(word + '\n' for word in all_words)

        for i in range(len(all_words)):
            self.word_index_dict[all_words[i]] = i
        print("Word index normalizing dict prepared.")

    def get_all_normalized_data_labels(self, use_narray=True):
        data = []
        labels = []  # one-hot labels
        for activity in self.activities:
            words = self.cut_activity_to_words(activity)
            normalized_word_count = np.zeros(len(self.word_index_dict)) if use_narray else [0] *len(self.word_index_dict)
            for word in words:
                index = self.word_index_dict[word]
                normalized_word_count[index] += 1
            data.append(normalized_word_count)
            labels.append(self.create_ont_hot_tags_label(activity['ActivityTags']))
        if use_narray:
            return np.array(data, dtype=np.int32), np.array(labels, dtype=np.int32)
        else:
            return data, labels

    @staticmethod
    def create_ont_hot_tags_label(activity_tags):
        one_hot = np.zeros(len(DataPreprocessor.ACTIVITY_TAGS))
        for tag in activity_tags:
            index = DataPreprocessor.ACTIVITY_TAGS.index(tag['name'])
            one_hot[index] = 1
        return one_hot

    def output(self, factory_method, filename):
        data, labels = self.get_all_normalized_data_labels(use_narray=False)
        for i in range(len(data)):
            activity_id = self.activities[i]['id']
            activity_title = self.activities[i]['title'].strip().replace("\n", "").replace("\r", "")
            activity_content = self.activities[i]['content'].strip().replace("\n", "").replace("\r", "")
            data[i] = [activity_id, activity_title, activity_content] + data[i]
        attrs = [('id', 'numeric'), ('title', 'string'), ('content', 'string')] + \
                [('v' + str(index), 'numeric') for index in range(len(self.word_index_dict))]
        text = factory_method('Activities', attrs, data)
        with open(filename, 'w+', encoding='utf-8') as fw:
            fw.write(text)

    def clustering(self, n_cluster=2,
                   n_showing_activities_each_cluster=20,
                   show_result=True,
                   output_file="clustering_result.json"):
        data, labels = self.get_all_normalized_data_labels()
        kmeans_fit = cluster.KMeans(n_clusters=n_cluster).fit(data)
        activities_in_clusters = [[] for i in range(n_cluster)]
        wordset_in_clusters = [set() for i in range(n_cluster)]

        cluster_labels = kmeans_fit.labels_
        for i in range(len(data)):
            index_cluster = cluster_labels[i]
            activities_in_clusters[index_cluster].append(self.activities[i])

        for i in range(n_cluster):
            wordset_in_clusters[i] = None
            for activity in activities_in_clusters[i]:
                words = set(self.cut_activity_to_words(activity))
                if not wordset_in_clusters[i]:
                    wordset_in_clusters[i] = words
                else:
                    wordset_in_clusters[i] &= words

        if show_result:
            for i in range(n_cluster):
                print("\n\nActivities in cluster " + str(i) + ", Count: " + str(len(activities_in_clusters[i])))
                print("Common words: ", wordset_in_clusters[i])
                for activity in activities_in_clusters[i][0:n_showing_activities_each_cluster]:
                    print(activity['id'], activity['title'].strip().replace('\n',''), activity['content'].strip().replace('\n',''))

        output_json_dict = {}
        for i in range(n_cluster):
            output_json_dict[i] = activities_in_clusters[i]

        with open(output_file, 'w+', encoding='utf-8') as fw:
            json.dump(output_json_dict, fw, ensure_ascii=False)

        return activities_in_clusters


preprocessor = DataPreprocessor()
preprocessor.start_preprocessing()
preprocessor.clustering(n_cluster=5)
