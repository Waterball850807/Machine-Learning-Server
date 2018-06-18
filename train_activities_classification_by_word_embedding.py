from keras.callbacks import ModelCheckpoint
import numpy as np


import model_utils
import DataPreprocessor
from keras.layers import *
from keras.models import *

if __name__ == '__main__':
    preprocessor = DataPreprocessor.DataPreprocessor()
    data, labels = preprocessor.get_words_embedding_preprocessing()

    random_mask = np.arange(len(data))
    np.random.shuffle(random_mask)

    count = len(data)
    data = data[random_mask]
    labels = labels[random_mask]

    print('Data shape: ', data.shape)
    print('Labels shape: ', labels.shape)

    epoch = 350
    batch_size = 30
    model = Sequential()
    model.add(Dense(1500, input_dim=data.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1500))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1500))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(labels.shape[1], activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    filepath_hdf5 = "word_embedding_model_e{}_b{}_750_3layers.hdf5".format(epoch, batch_size)
    checkpoint = ModelCheckpoint(filepath_hdf5)
    model.fit(data, labels,
              epochs=epoch,
              batch_size=batch_size,
              callbacks=[checkpoint])

    model_utils.torture_word_embedding_model(model)
