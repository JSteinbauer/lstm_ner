import math
import os
from abc import abstractmethod, ABCMeta
from enum import Enum
from typing import List, Dict, Generator, Any, Optional

import numpy as np
from tensorflow import Tensor

from ner.data_handling.data_batch import NerDataBatch


class DatasetName(Enum):
    """ Valid names of the datasets """
    TRAIN: str = 'train'
    VALID: str = 'valid'
    TEST: str = 'test'


class NerDataSiloBase(metaclass=ABCMeta):
    """
    Class to manage data for training/evaluation of NER Models.
    Upon initialization, it searches the data_directory for data files
    named '{train/valid/test}.words.txt' and '{train/valid/test}.tags.txt'
    """
    def __init__(
            self,
            data_directory: str,
            batch_size: int = 20,
            random_seed: Optional[int] = None,
            **kwargs: Any,
    ) -> None:
        self.data_directory = data_directory
        self.batch_size = batch_size

        self.data_dict: Dict[str, Any] = dict()
        # Fills data dict by reading data from data_directory
        self._fill_data_dict()
        # Optionally set random seed
        if random_seed:
            np.random.seed(random_seed)

    def _fill_data_dict(self) -> None:
        """ Try to fill self.data_dict by reading in data from directory """
        for dataset_name in DatasetName:
            try:
                word_lines: List[str] = self._read_lines(os.path.join(
                    self.data_directory, f'{dataset_name.value}.words.txt'))
                tag_lines: List[str] = self._read_lines(
                    os.path.join(self.data_directory, f'{dataset_name.value}.tags.txt'))
            except FileNotFoundError:
                pass
            else:
                self.data_dict[dataset_name.value] = (word_lines, tag_lines)

    @staticmethod
    def _read_lines(file_name: str) -> List[str]:
        """ Read data from file """
        with open(file_name) as f:
            return f.readlines()

    def get_dataset_size(self, dataset_name: str) -> int:
        """ Returns number of samples in dataset """
        return len(self.data_dict[dataset_name][0])

    def get_training_steps(self, epochs: int) -> int:
        """ Calculate and return the number of training steps """
        return epochs*math.ceil(self.get_dataset_size(dataset_name=DatasetName.TRAIN)/self.batch_size)

    def get_valid_steps(self, epochs: int = 1) -> int:
        """ Calculate and return the number of validation steps """
        return epochs*math.ceil(self.get_dataset_size(dataset_name=DatasetName.VALID)/self.batch_size)

    @abstractmethod
    def batch_generator(self, dataset_name: DatasetName) -> Generator:
        pass

    @abstractmethod
    def get_data_tensors(self, dataset_name: DatasetName) -> List[Tensor]:
        pass


class LstmNerDataSilo(NerDataSiloBase):
    """
    Class to manage data for training/evaluation of LSTM-based NER Models.
        - Applicable to models that consider character information, e.g. CharsConvLstmCrf (use_chars=True)
        - Applicable to models that operate on a word level only e.g. LstmCrf (use_chars=False)
    """
    def __init__(
            self,
            data_directory: str,
            batch_size: int = 20,
            use_chars: bool = False,
            random_seed: Optional[int] = None,
            **kwargs,
    ) -> None:
        super(LstmNerDataSilo, self).__init__(data_directory, batch_size, random_seed, **kwargs)
        # Whether or not to use character information
        self.use_chars = use_chars

    def batch_generator(self, dataset_name: str) -> Generator:
        """
        Generate data batches for LstmCrf (self.use_chars=False) or CharsConvLstmCrf (self.use_chars=True)
        Args:
            dataset_name: Name of the dataset
        Returns:
            A data generator
        """
        # Get requested dataset from data_dict
        word_lines, tag_lines = self.data_dict[dataset_name]

        counter = 0
        num_docs = len(word_lines)
        # Batch lists
        data_batch: NerDataBatch = NerDataBatch(use_chars=self.use_chars)

        while True:
            rand_index = np.random.permutation(num_docs)
            word_lines = [word_lines[ridx] for ridx in rand_index]
            tag_lines = [tag_lines[ridx] for ridx in rand_index]
            for line_words, line_tags in zip(word_lines, tag_lines):
                data_batch.add_sample(sample_phrase=line_words, sample_tags=line_tags)

                counter += 1
                if counter % self.batch_size == 0:
                    tensor_batch = data_batch.to_tensor()
                    yield tensor_batch, tensor_batch[-1]
                    # clear batch data
                    data_batch.clear_data()

    def get_data_tensors(self, dataset_name: DatasetName) -> List[Tensor]:
        """
        Processes the data of words_path and tags_path and returns it as tensors that can be
        directly fed into a model.
        """
        data_batch: NerDataBatch = NerDataBatch()
        for line_words, line_tags in zip(self.data_dict[dataset_name.value]):
            data_batch.add_sample(sample_phrase=line_words, sample_tags=line_tags)
        return data_batch.to_tensor()
