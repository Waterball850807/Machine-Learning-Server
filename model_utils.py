import DataPreprocessor
import words_preprocessing_utils
from word_vector_utils import get_word2vec_model, get_word_vector_size
import numpy as np
import pickle


def torture_rnn_model(model, max_words, word_list):
    while True:
        try:
            content = words_preprocessing_utils.clean_html(input('Content: '))
            x = words_preprocessing_utils.get_words_2d_representation(content, max_words,
                                                                      word_list,
                                                                      feature_type=words_preprocessing_utils.feature_word_embedding
                                                                      ) / DataPreprocessor.WORD_VECTOR_NORMALIZED_FACTOR

            y = model.predict(x)[0]
            print(y)
            sorted_indices = [i for i in np.argsort(y)[::-1][:3] if y[i] >= 0.5]  # get top 3 classifications
            if len(sorted_indices) == 0:
                print(DataPreprocessor.ACTIVITY_TAGS[np.argmax(y)])
            for index in sorted_indices:
                print(DataPreprocessor.ACTIVITY_TAGS[index])
        except Exception as e:
            print(e)


def torture_cnn_model(model, max_words, word_list):
    while True:
        try:
            content = words_preprocessing_utils.clean_html(input('Content: '))
            x = words_preprocessing_utils.get_words_2d_representation(content, max_words,
                                                                      word_list,
                                                                      feature_type=words_preprocessing_utils.feature_word_embedding
                                                                      ) / DataPreprocessor.WORD_VECTOR_NORMALIZED_FACTOR
            x = np.reshape(x, (1, 1, x.shape[0], x.shape[1]))
            y = model.predict(x)[0]
            print(y)
            sorted_indices = [i for i in np.argsort(y)[::-1][:3] if y[i] >= 0.5]  # get top 3 classifications
            if len(sorted_indices) == 0:
                print(DataPreprocessor.ACTIVITY_TAGS[np.argmax(y)])
            for index in sorted_indices:
                print(DataPreprocessor.ACTIVITY_TAGS[index])
        except Exception as e:
            print(e)


def torture_word_frequency_model(model, word_list):
    while True:
        try:
            content = words_preprocessing_utils.clean_html(input('Content: '))
            x = words_preprocessing_utils.get_word_frequency_vector(content,
                                                                    word_list) / DataPreprocessor.WORD_VECTOR_NORMALIZED_FACTOR
            x = np.reshape(x, (1, len(x)))
            y = model.predict(x)[0]
            print(y)
            sorted_indices = [i for i in np.argsort(y)[::-1][:3] if y[i] >= 0.5]  # get top 3 classifications
            if len(sorted_indices) == 0:
                print(DataPreprocessor.ACTIVITY_TAGS[np.argmax(y)])
            for index in sorted_indices:
                print(DataPreprocessor.ACTIVITY_TAGS[index])
        except Exception as e:
            print(e)


def torture_word_embedding_model(model):
    while True:
        try:
            content = words_preprocessing_utils.clean_html(input('Content: '))
            x = words_preprocessing_utils.get_word_embedding_vector(content)
            x = np.reshape(x, (1, len(x)))
            y = model.predict(x)[0]
            print(y)
            sorted_indices = [i for i in np.argsort(y)[::-1][:3] if y[i] >= 0.5]  # get top 3 classifications
            if len(sorted_indices) == 0:
                print(DataPreprocessor.ACTIVITY_TAGS[np.argmax(y)])
            for index in sorted_indices:
                print(DataPreprocessor.ACTIVITY_TAGS[index])
        except Exception as e:
            print(e)
