import json
import logging
import os
import time
from abc import abstractmethod, ABCMeta
from typing import Optional, Dict, Any, Generator, List

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from ner.data_handling.data_silos import LstmNerDataSilo, DatasetName
from ner.models.config_dataclasses import NerLstmConfig
from ner.utils.gpu import setup_strategy
from ner.utils.file_system import is_file, create_dir

log = logging.getLogger(__name__)


class NerGloveBase(metaclass=ABCMeta):
    """
    Base Class for all Glove embedding based Named Entity Recognition (NER) engines
    """
    PARAMS_FILENAME: str = 'params.json'
    WEIGHTS_FILENAME: str = 'weights.h5'

    def __init__(
            self,
            data_dir: str,
            model_dir: str,
            config: Optional[NerLstmConfig] = None,
            random_seed: Optional[int] = None,
            **kwargs,
    ) -> None:
        """
        Args:
            data_dir: Directory of the training data, model vocabulary and word embeddings
            model_dir: Directory to which model weights and training information is written
            config: Optional object holding the model and training configuration. If None,
                the default config is used.
            random_seed: Optional random seed to make training reproducible.
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.random_seed = random_seed

        # Define and validate resource directories
        self.vocab_words_dir: str = os.path.join(self.data_dir, 'vocab.words.txt')
        self.vocab_tags_dir: str = os.path.join(self.data_dir, 'vocab.tags.txt')
        self.glove_dir: str = os.path.join(self.data_dir, 'glove.npz')
        # Validate the existence of the directories defined above
        self._validate_resource_dirs()

        # Number of tags i.e. number of classification categories
        self.num_tags: int = self._get_num_tags()
        # If no params are parsed, use default params
        self.config: NerLstmConfig = config or NerLstmConfig.default_config()
        # Build the model
        self.keras_model = self.build_model(random_seed=random_seed)
        # Initialize data silo
        self.data_silo: Optional[LstmNerDataSilo] = None

    def _validate_resource_dirs(self):
        """ Validate the existence of relevant files """
        if not os.path.isfile(self.vocab_words_dir):
            raise FileNotFoundError('Path for words_file_path must be set and file must exist')

        if not os.path.isfile(self.vocab_tags_dir):
            raise FileNotFoundError('Path for tags_file_path must be set and file must exist')

        if not os.path.isfile(self.glove_dir):
            raise FileNotFoundError('Path for embedding_file_path must be set and file must exist')

    def _get_num_tags(self) -> int:
        """ Evaluates and returns the number of tags used for classification """
        with open(self.vocab_tags_dir) as f:
            indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
            return len(indices) + 1

    def save_model_config(self) -> None:
        """ save model configuration in form of params.json """
        with open(os.path.join(self.model_dir, self.PARAMS_FILENAME), 'w') as f:
            json.dump(self.config.to_dict(), f, indent=4, sort_keys=True)

    def model_summary(self) -> None:
        """ Print model summary """
        self.keras_model.summary()

    def build_model(self, random_seed: Optional[int] = None) -> tf.keras.Model:
        """
        Build and return the model.
        Args:
            random_seed: Optional integer to set the random seed for the initialization of the model layers.
        Returns:
            A keras model.
        """
        if random_seed:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

        with setup_strategy(num_gpus=self.config.num_gpus).scope():
            return self._build_model()

    @abstractmethod
    def _build_model(self) -> tf.keras.Model:
        """ Abstract method to be replaced with implementation of model. """
        pass

    def train(self, data_silo: Optional[LstmNerDataSilo] = None) -> None:
        """ Trains the model and stores the results in self.model_dir """
        if not (data_silo or self.data_silo):
            raise ValueError('Datasilo must be defined for training!')

        data_silo: Optional[LstmNerDataSilo] = data_silo or self.data_silo
        # Run training
        self._train(data_silo=data_silo)
        self.save_weights()

    def _train(self, data_silo: LstmNerDataSilo) -> None:
        """ Runs the training of the model """
        epochs: int = self.config.epochs

        n_train_samples: int = data_silo.get_dataset_size(DatasetName.VALID)
        n_valid_samples: int = data_silo.get_dataset_size(DatasetName.VALID)

        start_time: float = time.time()
        log.info(f'START Training {self.__class__.__name__} using {n_train_samples} samples '
                 f'in {epochs} epochs with validation using {n_valid_samples} samples...')

        # Set up the training data generator
        train_data: Generator = data_silo.batch_generator(DatasetName.TRAIN)
        train_steps: int = data_silo.get_training_steps_per_epoch()
        # Optionally set up the validation data generator
        valid_data: Optional[Generator] = None
        valid_steps: Optional[int] = None
        if DatasetName.VALID.value in data_silo.data_dict:
            valid_data = data_silo.batch_generator(DatasetName.VALID)
            valid_steps = data_silo.get_valid_steps_per_epoch()

        # Model is compiled without additional loss function.
        # Use loss added with add_loss() method
        self.keras_model.compile(loss=None, optimizer='adam')

        # set up callbacks
        callbacks: List[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir=self.model_dir),
        ]

        # Train the model
        self.keras_model.fit(
            train_data,
            validation_data=valid_data,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_steps=valid_steps,
            callbacks=callbacks,
        )

        log.info('DONE Training of %s in %f.2f s.', self.__class__.__name__, time.time() - start_time)

    def save_weights(self) -> None:
        """ Save weights of trained model """
        # Create model directory if it doesn't yet exist
        create_dir(self.model_dir)
        # Save weights
        self.keras_model.save_weights(filepath=os.path.join(self.model_dir, self.WEIGHTS_FILENAME))

    def load_weights(self, model_dir: str) -> None:
        """ Load weights of trained model """
        weights_path: str = os.path.join(model_dir, self.WEIGHTS_FILENAME)
        if not is_file(weights_path):
            raise FileNotFoundError(f'Weights file "{weights_path}" not found.')
        self.keras_model.load_weights(filepath=weights_path)

    def predict(self, model_input: list) -> np.ndarray:
        return self.keras_model.predict(model_input)

    def predict_on_file(
            self,
            name: str,
            language_model_dir_path: str,

    ) -> None:
        """ Apply trained model to data from file. Write predictions to another file.
        Args:
            name: Name of the data
            language_model_dir_path: Output directory for saving files
            data_dir: Input directory to read data from
            use_chars: Whether or not to use character information
        """
        language_model_prediction_path = os.path.join(language_model_dir_path, 'predictions')
        os.makedirs(language_model_prediction_path, exist_ok=True)

        score_file_path: str = os.path.join(language_model_prediction_path, f'score_{name}.preds.txt')

        test_data = self.data_silo.get_data_tensors(DatasetName.TEST)
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
