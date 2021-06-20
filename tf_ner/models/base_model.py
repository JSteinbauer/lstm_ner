import json
import logging
import math
import os
import time
from abc import abstractmethod, ABCMeta
from typing import Optional, Dict, Any, Generator, List

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from tf_ner.models.keras_data_generation import get_word_data_tensors, data_generator_words
from tf_ner.utils.gpu import setup_strategy
from tf_ner.utils.file_system import is_file, create_dir


log = logging.getLogger(__name__)


def get_steps_per_epoch(n_samples: int, batch_size: int) -> int:
    return int(math.ceil(n_samples / batch_size))


def fwords(name: str, data_dir: str) -> str:
    return os.path.join(data_dir, f'{name}.words.txt')


def ftags(name: str, data_dir: str) -> str:
    return os.path.join(data_dir, f'{name}.tags.txt')


class NerLstmBase(metaclass=ABCMeta):
    """
    Base Class for all LSTM-based Named Entity Recognition (NER) engines
    """
    PARAMS_FILENAME: str = 'params.json'
    WEIGHTS_FILENAME: str = 'weights.h5'
    DEFAULT_PARAMS: Dict[str, str] = {
        'dim_chars': 100,
        'learning_rate': 1e-3,
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 20,
        'batch_size': 100,
        'buffer': 15000,
        'filters': 50,
        'kernel_size': 3,
        'lstm_size': 100,
    }

    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        params: Optional[dict] = None,
        num_gpus: int = 0,
    ) -> None:
        if not data_dir and is_file(data_dir, exception=False):
            raise ValueError('Path for data directory must be set')
        self.data_dir = data_dir

        if not model_dir:
            raise ValueError('Path for model directory must be set')
        self.model_dir = model_dir

        # If no params are parsed, use default params
        self.params: Dict[str, Any] = params or self.DEFAULT_PARAMS
        # Add missing parameters
        self._add_missing_params()
        # create model directory
        create_dir(model_dir)

        # save model configuration in form of params.json
        with open(os.path.join(model_dir, self.PARAMS_FILENAME), 'w') as f:
            json.dump(self.params, f, indent=4, sort_keys=True)

        with setup_strategy(num_gpus=num_gpus).scope():
            self.keras_model = self._build_keras_model()

    def model_summary(self) -> None:
        """ Print model summary """
        self.keras_model.summary()

    def _add_missing_params(self) -> None:
        """
        Adds missing parameters
        """
        if 'words' not in self.params:
            words_file_path: str = os.path.join(self.data_dir, 'vocab.words.txt')
            if not words_file_path and is_file(words_file_path, exception=False):
                raise ValueError('Path for words_file_path must be set and file must exist')
            self.params['words'] = words_file_path

        if 'tags' not in self.params:
            tags_file_path = os.path.join(self.data_dir, 'vocab.tags.txt')
            if not tags_file_path and is_file(tags_file_path, exception=False):
                raise ValueError('Path for tags_file_path must be set and file must exist')
            self.params['tags'] = tags_file_path

        if 'chars' not in self.params:
            chars_file_path = os.path.join(self.data_dir, 'vocab.chars.txt')
            if not chars_file_path and is_file(chars_file_path, exception=False):
                raise ValueError('Path for chars_file_path must be set and file must exist')
            self.params['chars'] = chars_file_path

        if 'glove' not in self.params:
            glove_npz_file_path = os.path.join(self.data_dir, 'glove.npz')
            if not glove_npz_file_path and is_file(glove_npz_file_path, exception=False):
                raise ValueError('Path for glove_npz_file_path must be set and file must exist')
            self.params['glove'] = glove_npz_file_path

    @abstractmethod
    def _build_keras_model(self) -> tf.keras.Model:
        pass

    @abstractmethod
    def train(self, n_train_samples: int, n_valid_samples: int) -> None:
        pass

    def _train(
        self,
        n_train_samples: int,
        n_valid_samples: int,
        use_chars: bool,
    ) -> None:
        batch_size: int = self.params['batch_size']
        epochs: int = self.params['epochs']
        n_train_steps: int = get_steps_per_epoch(n_train_samples, batch_size=batch_size)
        n_valid_steps: int = get_steps_per_epoch(n_valid_samples, batch_size=batch_size)

        start_time: float = time.time()
        log.debug(f'START Training {self.__class__.__name__} using {n_train_samples} samples '
                  f'({n_train_steps} steps) in {epochs} epochs with validation using {n_valid_samples} '
                  f'samples ({n_valid_steps} steps)...')

        train_data_dir = fwords('train', self.data_dir)
        train_tags_dir = ftags('train', self.data_dir)
        valid_data_dir = fwords('testa', self.data_dir)
        valid_tags_dir = ftags('testa', self.data_dir)

        train_data: Generator = data_generator_words(train_data_dir, train_tags_dir,
                                                     batch_size=batch_size, use_chars=use_chars)
        valid_data: Optional[Generator] = None
        if n_valid_samples:
            valid_data = data_generator_words(valid_data_dir, valid_tags_dir,
                                              batch_size=batch_size, use_chars=use_chars)

        # model is compiled without additional loss function - only use loss added with add_loss() method
        self.keras_model.compile(loss=None, optimizer='adam')

        # set up callbacks
        callbacks: List[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir=self.model_dir),
        ]

        self.keras_model.fit(
            train_data,
            validation_data=valid_data,
            steps_per_epoch=n_train_steps,
            epochs=epochs,
            validation_steps=n_valid_steps,
            callbacks=callbacks,
        )

        log.debug('DONE Training of %s in %f.2f s.', self.__class__.__name__, time.time() - start_time)
        self.keras_model.save_weights(filepath=os.path.join(self.model_dir, self.WEIGHTS_FILENAME))

    def load_weights(self, model_dir: str) -> None:
        weights_path: str = os.path.join(model_dir, self.WEIGHTS_FILENAME)
        if not is_file(weights_path):
            raise FileNotFoundError(f'Weights file "{weights_path}" not found.')
        self.keras_model.load_weights(filepath=weights_path)

    def predict(self, model_input: list) -> np.ndarray:
        return self.keras_model.predict(model_input)

    def predict_test_file(
        self,
        name: str,
        language_model_dir_path: str,
        data_dir: str,
        use_chars: bool,
    ) -> None:
        """ Write predictions of trained model into files.
        Args:
            name: Name of the data
            language_model_dir_path: Output directory for saving files
            data_dir: Input directory to read data from
            use_chars: Whether or not to use character information
        """
        language_model_prediction_path = os.path.join(language_model_dir_path, 'predictions')
        os.makedirs(language_model_prediction_path, exist_ok=True)

        score_file_path: str = os.path.join(language_model_prediction_path, f'score_{name}.preds.txt')
        test_data_path = fwords(name, data_dir)
        test_tags_path = ftags(name, data_dir)

        # If there's no data to predict => skip
        if not os.path.getsize(test_data_path):
            return None

        test_data = get_word_data_tensors(test_data_path, test_tags_path, use_chars=use_chars)
        predictions = self.keras_model.predict(test_data)
        # Write predictions to file
        self.write_predictions(score_file_path, test_data, predictions)

    @staticmethod
    def write_predictions(file_dir: str, inputs: List[Tensor], predictions: Tensor):
        """
        Write predictions to file.
        Args:
            file_dir: Directory of output file to which predictions are written
            inputs: The input tensors used to make predictions
            predictions: The predicted labels
        """
        words, nwords, labels = inputs
        with open(file_dir, 'wb') as f:
            for n in range(predictions.shape[0]):
                for m in range(nwords[n]):
                    f.write(b' '.join([words[n, m].numpy(), labels[n, m].numpy(), predictions[n, m]]))
                    f.write(b'\n')
                f.write(b'\n')
