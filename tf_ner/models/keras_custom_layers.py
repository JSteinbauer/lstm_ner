from typing import Tuple, Dict, Any, List

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import Tensor
from tensorflow.python.ops.lookup_ops import index_table_from_file, index_to_string_table_from_file


# from tensorflow.python.types.core import Tensor


# ============================ Custom Layers ============================ #

class WordsToNumbers(tf.keras.layers.Layer):
    def __init__(self, param_words: str, param_tags: str, param_num_oov_buckets: int, **kwargs: Any) -> None:
        super(WordsToNumbers, self).__init__(**kwargs)
        self.param_words = param_words
        self.param_tags = param_tags
        self.param_num_oov_buckets = param_num_oov_buckets
        self.vocab_words = index_table_from_file(self.param_words, num_oov_buckets=self.param_num_oov_buckets)
        self.vocab_tags = index_table_from_file(self.param_tags)

    def call(self, sentence_tensor: Tensor, tag_string_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        token_tensor = self.vocab_words.lookup(sentence_tensor)
        tag_tensor = self.vocab_tags.lookup(tag_string_tensor)
        return token_tensor, tag_tensor

    def get_config(self) -> Dict:
        config: Dict = super(WordsToNumbers, self).get_config()
        config.update({'param_words': self.param_words,
                       'param_tags': self.param_tags,
                       'param_num_oov_buckets': self.param_num_oov_buckets})
        return config

class WordsCharsToNumbers(tf.keras.layers.Layer):
    def __init__(self, param_words: str, param_chars: str, param_tags: str, param_num_oov_buckets: int,
                 **kwargs: Any) -> None:
        super(WordsCharsToNumbers, self).__init__(**kwargs)
        self.param_words = param_words
        self.param_chars = param_chars
        self.param_tags = param_tags
        self.param_num_oov_buckets = param_num_oov_buckets

        self.vocab_words = index_table_from_file(self.param_words, num_oov_buckets=self.param_num_oov_buckets)
        self.vocab_chars = index_table_from_file(self.param_chars, num_oov_buckets=self.param_num_oov_buckets)
        self.vocab_tags = index_table_from_file(self.param_tags)

    def call(
        self,
        sentence_tensor: Tensor,
        char_tensor: Tensor,
        tag_string_tensor: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        token_tensor = self.vocab_words.lookup(sentence_tensor)
        char_id_tensor = self.vocab_chars.lookup(char_tensor)
        tag_tensor = self.vocab_tags.lookup(tag_string_tensor)
        return token_tensor, char_id_tensor, tag_tensor

    def get_config(self) -> Dict:
        config: Dict = super(WordsCharsToNumbers, self).get_config()
        config.update({'param_words': self.param_words,
                       'param_chars': self.param_chars,
                       'param_tags': self.param_tags,
                       'param_num_oov_buckets': self.param_num_oov_buckets})
        return config

class NumbersToTags(tf.keras.layers.Layer):
    def __init__(self, param_tags: str, **kwargs: Any) -> None:
        super(NumbersToTags, self).__init__(**kwargs)
        self.param_tags = param_tags
        self.reverse_vocab_tags = index_to_string_table_from_file(self.param_tags)

    def call(self, pred_ids: Tensor) -> Tensor:
        pred_strings = self.reverse_vocab_tags.lookup(tf.cast(pred_ids, tf.int64))
        return pred_strings

    def get_config(self) -> Dict:
        config: Dict = super(NumbersToTags, self).get_config()
        config.update({'param_tags': self.param_tags})
        return config


# Custom keras embedding layer using Glove
class WordsToEmbeddings(tf.keras.layers.Layer):
    def __init__(self, glove_path: str, embedding_dimension: int, **kwargs: Any) -> None:
        super(WordsToEmbeddings, self).__init__(**kwargs)

        self.glove_path = glove_path
        self.embedding_dimension = embedding_dimension
        self.glove = np.load(self.glove_path)['embeddings']
        self.variable = np.vstack([self.glove, [[0.] * self.embedding_dimension]])
        self.variable = tf.Variable(self.variable, dtype=tf.float32, trainable=False)

    def call(self, word_ids: Tensor) -> Tensor:
        embeddings = tf.nn.embedding_lookup(params=self.variable, ids=word_ids)
        return embeddings

    def get_config(self) -> Dict:
        config: Dict = super(WordsToEmbeddings, self).get_config()
        config.update({'glove_path': self.glove_path,
                       'embedding_dimension': self.embedding_dimension})
        return config


# Custom keras conditional random field layer
class CRFDecode(tf.keras.layers.Layer):
    def __init__(self, num_tags: int, **kwargs: Any) -> None:
        super(CRFDecode, self).__init__(**kwargs)
        self.num_tags = num_tags
        self.nwords = None

    def build(self, input_shape: List[Any]) -> None:
        assert len(input_shape) == 3, "input tensor must be of rank 3 in CRF_decode!"
        self.crf_params = self.add_weight(
            name="crf_weights",
            shape=[self.num_tags, self.num_tags],
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, logits: Tensor, nwords: Tensor, tags: Tensor) -> Tensor:
        self.nwords = nwords
        pred_ids, _ = tfa.text.crf_decode(logits, self.crf_params, self.nwords)

        # Calculate loss
        log_likelihood, _ = tfa.text.crf_log_likelihood(logits, tags, nwords, self.crf_params)
        loss = tf.reduce_mean(input_tensor=-log_likelihood)
        self.add_loss(loss)

        return pred_ids

    # Add loss for training
    def loss(self) -> Tensor:
        loss = self.losses[0]
        return loss

    def get_config(self) -> Dict:
        config: Dict = super(CRFDecode, self).get_config()
        config.update({"num_tags": self.num_tags})
        return config
