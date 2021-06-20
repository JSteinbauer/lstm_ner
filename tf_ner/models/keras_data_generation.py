from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Generator, Any

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.ops.lookup_ops import index_table_from_file


@dataclass
class NerDataBatch:
    """ Dataclass to hold the data of one batch """
    sentence_list: List[List[bytes]] = field(default_factory=list)
    nword_list: List[int] = field(default_factory=list)
    tag_list: List[List[bytes]] = field(default_factory=list)

    # Whether or not to use character information
    use_chars: bool = False
    char_list_list: List[List[List[bytes]]] = field(default_factory=list)
    char_length_list: List[List[int]] = field(default_factory=list)

    def clear_data(self) -> None:
        """ Resets all data lists to be empty """
        self.sentence_list = field(default_factory=list)
        self.nword_list = field(default_factory=list)
        self.char_list_list = field(default_factory=list)
        self.char_length_list = field(default_factory=list)
        self.tag_list = field(default_factory=list)

    def add_sample(self, sample_phrase: str, sample_tags: str) -> None:
        """
        Adds a sample to the batch.
        Args:
            sample_phrase: A phrase from which entities shall be extracted
            sample_tags: The corresponding entity tags
        """
        self._add_sample_words(sample_phrase)
        self._add_sample_tags(sample_tags)
        if self.use_chars:
            self._add_sample_chars(sample_phrase)

    def _add_sample_words(self, sample_phrase: str) -> None:
        """ Split sentences to obtain words """
        words = [w.encode() for w in sample_phrase.strip().split()]
        nwords = len(words)
        self.sentence_list.append(words)
        self.nword_list.append(nwords)

    def _add_sample_tags(self, sample_tags: str) -> None:
        """ Split tagged sentences """
        tags = [t.encode() for t in sample_tags.strip().split()]
        self.tag_list.append(tags)

    def _add_sample_chars(self, sample_phrase: str) -> None:
        """ Optionally add character information """
        chars = [[c.encode() for c in w] for w in sample_phrase.strip().split()]
        self.char_list_list.append(chars)
        self.char_length_list.append([len(c) for c in chars])

    def _apply_padding(self) -> None:
        """ Applies padding to self.sentence_list, self.tag_list and (optionally) self.char_list_list """

        # Pad self.sentence_list and self.tag_list
        max_nwords = max(self.nword_list)
        for n, (sentence, tag) in enumerate(zip(self.sentence_list, self.tag_list)):
            self.sentence_list[n] = sentence + (max_nwords - len(sentence)) * [b'<pad>']
            self.tag_list[n] = tag + (max_nwords - len(tag)) * [b'O']

        # Optionally pad self.char_list_list
        if self.use_chars:
            max_chars = max([max(c) for c in self.char_length_list])
            for n, (char_list, char_length) in enumerate(zip(self.char_list_list, self.char_length_list)):
                self.char_list_list[n] = [word + (max_chars - len(word)) * [b'<pad>'] for word in char_list] + (
                        max_nwords - len(char_list)) * [[b'<pad>'] * max_chars]
                self.char_length_list[n] = char_length + (max_nwords - len(char_length)) * [0]

    def to_tensor(self) -> List[Tensor]:
        """ Transform data lists to tensors """
        self._apply_padding()

        sentence_tensor = tf.convert_to_tensor(self.sentence_list, tf.string)
        nwords_tensor = tf.convert_to_tensor(self.nword_list, tf.int32)
        tag_string_tensor = tf.convert_to_tensor(self.tag_list, tf.string)

        if self.use_chars:
            char_tensor = tf.convert_to_tensor(self.char_list_list, tf.string)
            nchar_tensor = tf.convert_to_tensor(self.char_length_list, tf.int32)

            return [sentence_tensor, nwords_tensor, char_tensor, nchar_tensor, tag_string_tensor]
        return [sentence_tensor, nwords_tensor, tag_string_tensor]

class NerDataSiloBase:
    def __init__(self, words_path: str, tags_path: str, batch_size: int = 20, **kwargs: Any) -> None:
        self.words_path = words_path
        self.tags_path = tags_path
        self.batch_size = batch_size

        self.word_lines: List[str] = self._read_lines(words_path)
        self.tag_lines: List[str] = self._read_lines(tags_path)

    @staticmethod
    def _read_lines(file_name: str) -> List[str]:
        with open(file_name) as f:
            return f.readlines()


class LstmNerDataSilo(NerDataSiloBase):
    """
    Class to manage data for training of LSTM-based NER Models.
        - Applicable to models that consider character information, e.g. CharsConvLstmCrf (use_chars=True)
        - Applicable to models that operate on a word level only e.g. LstmCrf (use_chars=False)
    """

    def __init__(self, words_path: str, tags_path: str, batch_size: int = 20, use_chars: bool = True, **kwargs) -> None:
        super(LstmNerDataSilo, self).__init__(words_path, tags_path, batch_size, **kwargs)
        # Whether or not to use character information
        self.use_chars = use_chars

    def data_generator_words(self) -> Generator:
        """ Generate data batches for LstmCrf (use_chars=False) + CharsConvLstmCrf (use_chars=True) """

        generator_idx = 0
        num_docs = len(self.word_lines)
        # Batch lists
        data_batch: NerDataBatch = NerDataBatch(use_chars=self.use_chars)

        while True:
            rand_index = np.random.permutation(num_docs)
            word_lines = [self.word_lines[ridx] for ridx in rand_index]
            tag_lines = [self.tag_lines[ridx] for ridx in rand_index]
            for line_words, line_tags in zip(word_lines, tag_lines):
                data_batch.add_sample(sample_phrase=line_words, sample_tags=line_tags)

                generator_idx += 1
                if generator_idx % self.batch_size == 0:
                    tensor_batch = data_batch.to_tensor()
                    yield tensor_batch, tensor_batch[-1]
                    # clear batch data
                    data_batch.clear_data()


def phrase_to_model_input(phrase: str, use_chars: bool = False) -> List:
    """
    Args:
        phrase: text phrase of which entities shall be extracted
        use_chars: set to True if characters shall be considered.

    Returns:
        List of tensors, if use_chars is True, len(output) = 5, else len(output)=3
    """
    words = phrase.split()
    nwords = len(words)
    words_tensor = tf.convert_to_tensor([words], tf.string)
    nwords_tensor = tf.convert_to_tensor([nwords], tf.int32)
    tags_tensor = tf.convert_to_tensor([['O'] * nwords], tf.string)

    if not use_chars:
        return [words_tensor, nwords_tensor, tags_tensor]

    else:
        nchars = [len(word) for word in words]
        max_num_chars = max(nchars)
        chars = [[c for c in word] + ['<pad>'] * (max_num_chars - len(word)) for word in words]

        chars_tensor = tf.convert_to_tensor([chars], tf.string)
        nchars_tensor = tf.convert_to_tensor([nchars], tf.int32)
        return [words_tensor, nwords_tensor, chars_tensor, nchars_tensor, tags_tensor]


def transform_list_to_tensors(data_lists: List, use_chars: bool = False) -> List:
    """ data preparation LstmCrf + CharsConvLstmCrf """
    if not use_chars:
        sentence_list, nword_list, tag_list = data_lists
    else:
        sentence_list, nword_list, char_list_list, char_length_list, tag_list = data_lists

    max_nwords = max(nword_list)

    for n, (sentence, tag) in enumerate(zip(sentence_list, tag_list)):
        sentence_list[n] = sentence + (max_nwords - len(sentence)) * ['<pad>']
        tag_list[n] = tag + (max_nwords - len(tag)) * [b'O']

    sentence_tensor = tf.convert_to_tensor(sentence_list, tf.string)
    nwords_tensor = tf.convert_to_tensor(nword_list, tf.int32)
    tag_string_tensor = tf.convert_to_tensor(tag_list, tf.string)

    if not use_chars:
        return [sentence_tensor, nwords_tensor, tag_string_tensor]
    else:
        max_chars = max([max(c) for c in char_length_list])
        for n, (char_list, char_length) in enumerate(zip(char_list_list, char_length_list)):
            char_list_list[n] = [word + (max_chars - len(word)) * [b'<pad>'] for word in char_list] + (
                    max_nwords - len(char_list)) * [[b'<pad>'] * max_chars]
            char_length_list[n] = char_length + (max_nwords - len(char_length)) * [0]

        char_tensor = tf.convert_to_tensor(char_list_list, tf.string)
        nchar_tensor = tf.convert_to_tensor(char_length_list, tf.int32)

        return [sentence_tensor, nwords_tensor, char_tensor, nchar_tensor, tag_string_tensor]


def get_data_lists(words_path: str, tags_path: str, use_chars: bool = False) -> List:
    sentence_list = []
    nword_list = []
    tag_list = []
    char_list_list = []
    char_length_list = []

    with Path(words_path).open('r') as f_words, Path(tags_path).open('r') as f_tags:
        for line_words in f_words:
            words = [w.encode() for w in line_words.strip().split()]
            nwords = len(words)
            sentence_list.append(words)
            nword_list.append(nwords)

            chars = [[c.encode() for c in w] for w in line_words.strip().split()]
            char_list_list.append(chars)
            char_length_list.append([len(c) for c in chars])

        for line_tags in f_tags:
            tags = [t.encode() for t in line_tags.strip().split()]
            tag_list.append(tags)

    if use_chars:
        return [sentence_list, nword_list, char_list_list, char_length_list, tag_list]
    else:
        return [sentence_list, nword_list, tag_list]


def get_numeric_data_tensors(word_path: str, tag_path: str, params: Dict, use_chars: bool = False) -> List:
    word_tag_data_lists = get_data_lists(word_path, tag_path, use_chars=use_chars)
    data_tensors = transform_list_to_tensors(
        word_tag_data_lists, use_chars=use_chars)

    vocab_words = index_table_from_file(params['words'], num_oov_buckets=params['num_oov_buckets'])
    vocab_tags = index_table_from_file(params['tags'])

    token_tensor = vocab_words.lookup(data_tensors[0])
    tag_tensor = vocab_tags.lookup(data_tensors[-1])

    if use_chars:
        vocab_chars = index_table_from_file(params['chars'], num_oov_buckets=params['num_oov_buckets'])
        char_id_tensor = vocab_chars.lookup(data_tensors[2])
        return [token_tensor, data_tensors[1], char_id_tensor, data_tensors[3], tag_tensor]
    else:
        return [token_tensor, data_tensors[1], tag_tensor]


def get_word_data_tensors(word_path: str, tag_path: str, use_chars: bool = False) -> List:
    word_tag_data_lists = get_data_lists(word_path, tag_path, use_chars=use_chars)
    data_tensors = transform_list_to_tensors(word_tag_data_lists, use_chars=use_chars)

    return data_tensors
