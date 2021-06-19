"""
GloVe Embeddings + chars conv and max pooling + bi-LSTM + CRF
Implementation based on https://arxiv.org/pdf/1508.01991.pdf
"""
import os.path

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.python.keras.layers import Dropout, Bidirectional, LSTM, Dense
from tensorflow.python.keras.models import Model

from tf_ner.models.base_model import NerBase
from tf_ner.models.keras_custom_layers import (
    WordsToNumbers,
    WordsToEmbeddings,
    CRFDecode,
    NumbersToTags,
)
from tf_ner.models.keras_data_generation import get_word_data_tensors


class NerLstmCrf(NerBase):
    def _build_keras_model(self) -> tf.keras.Model:
        tags_path: str = self.params['tags']
        with open(tags_path) as f:
            indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
            num_tags = len(indices) + 1

        # Inputs -> beware that tag labels are also model inputs, since they are needed for training
        input1 = Input(shape=(None,), dtype='string')
        input2 = Input(shape=(), dtype='int32')
        input3 = Input(shape=(None,), dtype='string')

        tokens, tags = WordsToNumbers(
            param_words=self.params['words'],
            param_tags=tags_path,
            param_num_oov_buckets=self.params['num_oov_buckets'],
        )(input1, input3)
        embeddings = WordsToEmbeddings(
            glove_path=self.params['glove'],
            embedding_dimension=self.params['dim'],
        )(tokens)
        dropout_rate: float = self.params['dropout']
        embeddings = Dropout(rate=dropout_rate)(embeddings)
        output = Bidirectional(LSTM(units=self.params['lstm_size'], return_sequences=True))(embeddings)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(units=num_tags)(output)
        output = CRFDecode(num_tags=num_tags)(output, input2, tags)
        output = NumbersToTags(param_tags=tags_path)(output)

        model = Model([input1, input2, input3], output)
        return model

    def train(self, n_train_samples: int, n_valid_samples: int) -> None:
        super()._train(
            n_train_samples=n_train_samples,
            n_valid_samples=n_valid_samples,
            use_chars=False,
        )

if __name__ == '__main__':
    data_dir: str = 'data/conll-2003_preprocessed/'
    model_dir: str = 'models/lstm_crf/'
    ner_lstm_crf = NerLstmCrf(
        data_dir=data_dir,
        model_dir=model_dir,
    )
    # test_data_path = os.path.join(data_dir, 'testb.words.txt')
    # test_tags_path = os.path.join(data_dir, 'testb.tags.txt')
    #
    # test_data = get_word_data_tensors(test_data_path, test_tags_path)
    # ner_lstm_crf.predict()
    ner_lstm_crf.train(14041, 3250)