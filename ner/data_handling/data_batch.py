from dataclasses import dataclass, field
from typing import List

import tensorflow as tf
from tensorflow import Tensor


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
