from pathlib import Path
from typing import List, Dict, Generator

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.lookup_ops import index_table_from_file


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


def data_generator_words(words_path: str, tags_path: str, batch_size: int = 20, use_chars: bool = True) -> Generator:
    """ Generate data batches for LstmCrf (use_chars=False) + CharsConvLstmCrf (use_chars=True) """
    with open(words_path) as f:
        word_lines = f.readlines()
    with open(tags_path) as f:
        tag_lines = f.readlines()

    idx = 0
    num_docs = len(word_lines)
    # Batch lists
    sentence_list = []
    nword_list = []
    char_list_list = []
    char_length_list = []
    tag_list = []

    while True:
        rand_index = np.random.permutation(num_docs)
        word_lines = [word_lines[ridx] for ridx in rand_index]
        tag_lines = [tag_lines[ridx] for ridx in rand_index]
        for line_words, line_tags in zip(word_lines, tag_lines):
            # Split sentences to obtain words
            words = [w.encode() for w in line_words.strip().split()]
            nwords = len(words)
            sentence_list.append(words)
            nword_list.append(nwords)

            if use_chars:
                chars = [[c.encode() for c in w] for w in line_words.strip().split()]
                char_list_list.append(chars)
                char_length_list.append([len(c) for c in chars])

            # Split tagged sentences
            tags = [t.encode() for t in line_tags.strip().split()]
            tag_list.append(tags)

            idx += 1
            if idx % batch_size == 0:
                if use_chars:
                    to_tensors = [sentence_list, nword_list, char_list_list, char_length_list, tag_list]
                else:
                    to_tensors = [sentence_list, nword_list, tag_list]

                tensor_batch = transform_list_to_tensors(
                    to_tensors, use_chars=use_chars)

                # empty batch lists
                sentence_list = []
                nword_list = []
                tag_list = []
                char_list_list = []
                char_length_list = []

                yield tensor_batch, tensor_batch[-1]


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
