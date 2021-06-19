from typing import Dict

import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import Model

from ondewo_cdls.extractors.tf_ner.models.keras_custom_layers import WordsToEmbeddings, CRFDecode
from ondewo_cdls.extractors.tf_ner.models.masked_conv import masked_conv1d_and_max


def build_keras_model_lstm_crf(params: Dict) -> Model:
    with open(params['tags']) as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    # Inputs -> beware that tag labels are also model inputs, since they are needed for training
    input1 = Input(shape=(None,), dtype='int32')
    input2 = Input(shape=(), dtype='int32')
    input3 = Input(shape=(None,), dtype='int32')

    embeddings = WordsToEmbeddings(params['glove'], params['dim'])(input1)
    embeddings = Dropout(params['dropout'])(embeddings)
    output = Bidirectional(LSTM(params['lstm_size'], return_sequences=True))(embeddings)
    output = Dropout(params['dropout'])(output)
    output = Dense(num_tags)(output)
    crf = CRFDecode(num_tags)
    output = crf(output, input2, input3)

    model = Model([input1, input2, input3], output)
    return model


def build_keras_model_chars_conv_lstm_crf(params: Dict) -> Model:
    with open(params['tags']) as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    with open(params['chars']) as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']

    input1 = Input(shape=(None,), dtype='int32')
    input2 = Input(shape=(), dtype='int32')
    input3 = Input(shape=(None, None), dtype='int32')
    input4 = Input(shape=(None,), dtype='int32')
    input5 = Input(shape=(None,), dtype='int32')

    char_embeddings = Embedding(num_chars, params['dim_chars'])(input3)
    char_embeddings = Dropout(params['dropout'])(char_embeddings)

    char_embeddings_shape = tf.shape(char_embeddings)
    mask_length = tf.reshape(tf.slice(char_embeddings_shape, [2], [1]), [])
    weights = tf.sequence_mask(input4, mask_length)
    char_embeddings = masked_conv1d_and_max(char_embeddings, weights, params['filters'],
                                            params['kernel_size'])

    word_embeddings = WordsToEmbeddings(params['glove'], params['dim'])(input1)

    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
    embeddings = Dropout(params['dropout'])(embeddings)
    output = Bidirectional(LSTM(params['lstm_size'], return_sequences=True))(embeddings)
    output = Dropout(params['dropout'])(output)
    pred_ids = Dense(num_tags)(output)
    crf = CRFDecode(num_tags)
    output = crf(pred_ids, input2, input5)

    model = Model([input1, input2, input3, input4, input5], output)
    return model


# Not working yet...
# class LstmCrfModel(tf.keras.Model):
#     def __init__(self, params):
#         super(LstmCrfModel, self).__init__()
#
#         with open(params['tags']) as f:
#             indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
#             num_tags = len(indices) + 1
#
#         self.wordembeddings = WordsToEmbeddings(params['glove'], params['dim'])
#         self.bidirectionalLSTM = Bidirectional(LSTM(params['lstm_size'], return_sequences=True))
#         self.dense = Dense(num_tags)
#         self.dropout = Dropout(params['dropout'])
#         self.crflayer = CRFDecode(num_tags)
#
#     def call(self, inputs, training=False):
#         input1 = inputs[0]
#         input2 = inputs[1]
#         input3 = inputs[2]
#         embeddings = self.wordembeddings(input1)
#         if training:
#             embeddings = self.dropout(embeddings)
#         output = self.bidirectionalLSTM(embeddings)
#         if training:
#             output = self.dropout(output)
#         output = self.dense(output)
#         output = self.crflayer(output, input2, input3)
#
#         return output
