from keras.callbacks import ModelCheckpoint
import model_utils
import DataPreprocessor
from keras.layers import *
from keras.models import *

if __name__ == '__main__':
    preprocessor = DataPreprocessor.DataPreprocessor()
    data, labels, word_list = preprocessor.get_cnn_representation_training_data_labels(
        DataPreprocessor.DataPreprocessor.FEATURE_TYPE_WORD_EMBEDDING)

    n_samples, channel, height, width = data.shape
    # the shape tensorflow supports is (samples, height, width, channel)
    data = np.reshape(data, (n_samples, height, width, channel))

    random_mask = np.arange(len(data))
    np.random.shuffle(random_mask)

    data = data[random_mask]
    labels = labels[random_mask]

    print('Data shape: ', data.shape)
    print('Labels shape: ', labels.shape)

    epoch = 1
    batch_size = 32
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(data.shape[1], data.shape[2], 1), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(DataPreprocessor.ACTIVITY_TAGS), activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    filepath_hdf5 = "cnn_word_frequency_model_e{}_b{}_3layers.hdf5".format(epoch, batch_size)
    checkpoint = ModelCheckpoint(filepath_hdf5)
    model.fit(data, labels,
              epochs=epoch,
              batch_size=batch_size,
              callbacks=[checkpoint])

    weights = model.get_weights()

    model_utils.torture_cnn_model(model, DataPreprocessor.MAX_WORD_COUNT, word_list)
