import re

import jieba_utils
import numpy as np

from word_vector_utils import *


def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def get_word_frequency_vector(content, word_list, cut_all=False):
    word_to_index = dict((word, index) for index, word in enumerate(word_list))
    wv = np.zeros(len(word_list) + 1)  # additional 1 for other words
    words = jieba_utils.cut(content, cut_all=cut_all)
    for word in words:
        if word in word_to_index:
            index = word_to_index[word]
            wv[index] += 1
        else:
            wv[-1] += 1
    return wv


def get_word_frequency_vectors(samples, words_file_name=None, output_file_name=None, cut_all=False):
    """
    :param samples: all samples with each sample is a text
    :param words_file_name: a file name which the file contains words in each line for reproducibility of word-to-index
    :param cut_all: cut_all to jieba.cut
    :return: (data, wordset, word_to_index), data is the result of getting word frequency vectors (numpy array),
        wordset is a set with all words from samples, word_to_index is a dict whose key is a word and the value is its index
    """
    print('Getting word frequency vectors...')
    wordset = set()
    words_results = []
    for sample in samples:
        words = jieba_utils.cut(sample, cut_all=cut_all)
        words_results.append(words)
        wordset.update(words)

    print('Words count: ', len(wordset))
    if words_file_name:
        print('Loading word list from file ', words_file_name)
        with open(words_file_name, 'r', encoding='utf-8') as fr:
            word_list = [line.strip() for line in fr.readlines() if len(line.strip()) != 0]
    else:
        print('Using random new word list.')
        word_list = [word.strip() for word in list(wordset) if len(word.strip()) != 0]

    if output_file_name:
        print('Outputting word list to file ', output_file_name)
        with open(output_file_name, 'w+', encoding='utf-8') as fw:
            for word in word_list:
                fw.write(word + '\n')

    word_to_index = dict((word, index) for index, word in enumerate(word_list))
    data = np.zeros((len(samples), len(word_list)+1))
    for i in range(len(words_results)):
        for word in words_results[i]:
            if word in word_to_index:
                data[i][word_to_index[word]] += 1
            else:
                data[i][-1] += 1

    print('Word frequency vectors loaded.')
    return data, word_list, word_to_index


def get_word_embedding_vector(content):
    word_model = get_word2vec_model()
    word_vector = np.zeros(get_word_vector_size())
    for word in jieba_utils.cut(content):
        if word in word_model.wv:
            word_vector += word_model.wv[word]
    return word_vector


def get_word_embedding_vectors(samples):
    data = []
    for text in samples:
        data.append(get_word_embedding_vector(text))
    return np.array(data)


if __name__ == '__main__':
    test_samples = ['哈哈哈你人真好', '老師你人很好', '妳很靠北欸', '老師老師快理我']
    results, word_list, word_to_index = get_word_frequency_vectors(test_samples,
                                                                   words_file_name='test.txt')

    print(word_to_index)
    print(results)