import os
from copy import copy
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.python.keras.layers import Dropout, Bidirectional, LSTM, Dense
from tensorflow.python.keras.models import Model

from ner.data_handling.data_silos import LstmNerDataSilo
from ner.data_handling.helper import phrase_to_model_input
from ner.models.base_model import NerGloveBase
from ner.models.config_dataclasses import NerLstmConfig
from ner.models.custom_layers import (
    WordsToNumbers,
    WordsToEmbeddings,
    CRFDecode,
    NumbersToTags,
)


class NerLstmCrf(NerGloveBase):
    """
    GloVe Embeddings + chars conv and max pooling + bi-LSTM + CRF
    Implementation based on https://arxiv.org/pdf/1508.01991.pdf
    """

    def __init__(
            self,
            data_dir: str,
            model_dir: str,
            config: Optional[NerLstmConfig] = None,
            random_seed: Optional[int] = None,
            **kwargs,
    ) -> None:
        super(NerLstmCrf, self).__init__(data_dir, model_dir, config, **kwargs)
        # Set up data silo for training
        self.data_silo: LstmNerDataSilo = LstmNerDataSilo(
            data_directory=data_dir,
            batch_size=self.config.batch_size,
            use_chars=False,
            random_seed=random_seed,
        )

    def _build_model(self) -> tf.keras.Model:
        """ Build the LSTM + CRF model. """
        # Inputs -> beware that tag labels are also model inputs, since they are needed for training
        input1 = Input(shape=(None,), dtype='string')
        input2 = Input(shape=(), dtype='int32')
        input3 = Input(shape=(None,), dtype='string')

        tokens, tags = WordsToNumbers(
            param_words=self.vocab_words_dir,
            param_tags=self.vocab_tags_dir,
            param_num_oov_buckets=self.config.num_oov_buckets,
        )([input1, input3])
        embeddings = WordsToEmbeddings(
            glove_path=self.glove_dir,
            embedding_dimension=self.config.embedding_dim,
        )(tokens)
        dropout_rate: float = self.config.dropout
        embeddings = Dropout(rate=dropout_rate)(embeddings)
        output = Bidirectional(LSTM(units=self.config.lstm_size, return_sequences=True))(embeddings)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(units=self.num_tags)(output)

        # Note that the loss function is defined in the CRFDecode layer
        output = CRFDecode(num_tags=self.num_tags)([output, input2, tags])
        output = NumbersToTags(param_tags=self.vocab_tags_dir)(output)

        model = Model([input1, input2, input3], output)
        return model


def run_model(int_seed: int):
    seed(int_seed)
    set_seed(int_seed)

    data_dir: str = 'data/conll-2003_preprocessed/'
    model_dir: str = 'models/lstm_crf/'
    ner_lstm_crf = NerLstmCrf(
        data_dir=data_dir,
        model_dir=model_dir,
    )
    ner_lstm_crf.keras_model.compile()
    # test_data_path = os.path.join(data_dir, 'test.words.txt')
    # test_self.vocab_tags_dir = os.path.join(data_dir, 'test.tags.txt')
    #
    # test_data = get_word_data_tensors(test_data_path, test_self.vocab_tags_dir)
    # ner_lstm_crf.predict()
    model_input = phrase_to_model_input('Tonight I will go shopping with Paris')
    prediction = ner_lstm_crf.predict(model_input)
    print(prediction)


if __name__ == '__main__':
    # from numpy.random import seed
    # from tensorflow.random import set_seed
    #
    # run_model(1)
    # run_model(1)
    # run_model(1)
    # run_model(3)
    # run_model(3)

    data_dir: str = 'data/conll-2003_preprocessed/'
    model_dir: str = 'models/lstm_crf/'
    ner_lstm_crf = NerLstmCrf(
        data_dir=data_dir,
        model_dir=model_dir,
    )
    ner_lstm_crf.model_summary()
    ner_lstm_crf.keras_model.compile()

    weights = copy(ner_lstm_crf.keras_model.trainable_weights)

    test_example = phrase_to_model_input(' '.join(['John']*50))
    ner_lstm_crf.keras_model.fit(x=test_example, y=test_example[2], batch_size=1, epochs=10)
    weights2 = copy(ner_lstm_crf.keras_model.trainable_weights)

    print('asdf')
    # test_data_path = os.path.join(data_dir, 'test.words.txt')
    # test_self.vocab_tags_dir = os.path.join(data_dir, 'test.tags.txt')
    #
    # test_data = get_word_data_tensors(test_data_path, test_self.vocab_tags_dir)
    # ner_lstm_crf.predict()
    # model_input = phrase_to_model_input('Tonight I will go shopping with Paris')
    # prediction = ner_lstm_crf.predict(model_input)
    # print(prediction)
    ner_lstm_crf.train()
