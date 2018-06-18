from keras.callbacks import ModelCheckpoint
import model_utils
import DataPreprocessor
from keras.layers import *
from keras.models import *

if __name__ == '__main__':
    preprocessor = DataPreprocessor.DataPreprocessor()
    data, labels, word_list = preprocessor.get_cnn_representation_training_data_labels(
        DataPreprocessor.DataPreprocessor.FEATURE_TYPE_WORD_EMBEDDING)
    n_samples, channel, time_steps, feature_size = data.shape
    data = np.reshape(data, (n_samples, time_steps, feature_size))

    random_mask = np.arange(len(data))
    np.random.shuffle(random_mask)

    data = data[random_mask]
    labels = labels[random_mask]

    print('Data shape: ', data.shape)
    print('Labels shape: ', labels.shape)

    epoch = 1
    batch_size = 32
    model = Sequential()
    model.add(LSTM(300, input_shape=(time_steps, feature_size), return_sequences=True))
    model.add(LSTM(300))
    model.add(Dense(len(DataPreprocessor.ACTIVITY_TAGS), activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    filepath_hdf5 = "rnn_word_frequency_model_e{}_b{}_3layers.hdf5".format(epoch, batch_size)
    checkpoint = ModelCheckpoint(filepath_hdf5)
    model.fit(data, labels,
              epochs=epoch,
              batch_size=batch_size,
              callbacks=[checkpoint])

    model_utils.torture_rnn_model(model, DataPreprocessor.MAX_WORD_COUNT, word_list)
