
from keras.layers import *
from keras.models import *
from words_preprocessing_utils import get_word_frequency_vector
from model_utils import torture_word_frequency_model
import DataPreprocessor

preprocessor = DataPreprocessor.DataPreprocessor()
data, labels, word_list = preprocessor.get_words_frequency()
model = load_model('word_frequency_model_e100_b30_200_3layers.hdf5')
weights = model.get_weights()
model.summary()

print(preprocessor.word_to_index['白目'])
print(preprocessor.word_to_index['申請書'])
print(preprocessor.word_to_index['無悔'])
print(preprocessor.word_to_index['當代'])
torture_word_frequency_model(model, word_list)


