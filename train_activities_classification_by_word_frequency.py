from keras.callbacks import ModelCheckpoint
import numpy as np


import model_utils
import DataPreprocessor
from keras.layers import *
from keras.models import *

import words_preprocessing_utils
from WeightsSaver import WeightsSaver

if __name__ == '__main__':
    preprocessor = DataPreprocessor.DataPreprocessor()

    data, labels, word_list = preprocessor.get_words_frequency()

    random_mask = np.arange(len(data))
    np.random.shuffle(random_mask)

    count = len(data)
    data = data[random_mask]
    labels = labels[random_mask]

    print('Data shape: ', data.shape)
    print('Labels shape: ', labels.shape)

    epoch = 300
    batch_size = 30
    model = Sequential()
    model.add(Dense(5000, input_dim=data.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5000))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5000))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(labels.shape[1], activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    filepath_hdf5 = "word_frequency_model_e{}_b{}_5000_3layers.hdf5".format(epoch, batch_size)
    checkpoint = ModelCheckpoint(filepath_hdf5)
    model.fit(data, labels,
              epochs=epoch,
              batch_size=batch_size,
              callbacks=[checkpoint])

    weights = model.get_weights()

    model_utils.torture_word_frequency_model(model, word_list)
