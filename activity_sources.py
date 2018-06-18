import json
import numpy as np


SOURCES_FILE_NAME = 'source_name_list.txt'

source_to_index_cache = None


def get_source_name_one_hot(source_name):
    source_name_to_index = load_source_name_to_index()
    one_hot = np.zeros(len(source_name_to_index))
    one_hot[source_name_to_index[source_name]] = 1
    return one_hot


def load_source_name_to_index():
    global source_to_index_cache
    if not source_to_index_cache:
        print('Loading source names...')
        with open(SOURCES_FILE_NAME, 'r', encoding='utf-8') as fr:
            source_names = [line.strip() for line in fr.readlines() if len(line.strip()) != 0]
            source_to_index_cache = dict((name, index) for index, name in enumerate(source_names))
    return source_to_index_cache


def save_all_source_names():
    with open('activities.json', 'r', encoding='utf-8') as fr:
        activities = json.load(fr)['data']

        source_name_set = set()
        for activity in activities:
            source_name_set.add(activity['source'])

        source_name_list = list(source_name_set)
        with open(SOURCES_FILE_NAME, 'w+', encoding='utf-8') as fw:
            for sn in source_name_list:
                fw.write(sn + '\n')
