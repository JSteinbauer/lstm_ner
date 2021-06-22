import math

import pytest

from ner.data_handling.data_silos import LstmNerDataSilo, DatasetName


class TestDataSilos:
    DATA_DIRECTORY: str = 'tests/resources'

    @staticmethod
    @pytest.mark.parametrize('batch_size', [
        1, 2, 5
    ])
    def test_ner_data_silo_base(batch_size: int):
        test_data_silo: LstmNerDataSilo = LstmNerDataSilo(
            data_directory=TestDataSilos.DATA_DIRECTORY,
            batch_size=batch_size,
        )

        train_data_size: int = test_data_silo.get_dataset_size(DatasetName.TRAIN)
        valid_data_size: int = test_data_silo.get_dataset_size(DatasetName.VALID)

        assert test_data_silo.get_training_steps_per_epoch() == math.ceil(train_data_size/batch_size)
        assert test_data_silo.get_valid_steps_per_epoch() == math.ceil(valid_data_size/batch_size)

        # Check that the returned batch size is correct
        for batch in test_data_silo.batch_generator(DatasetName.TRAIN):
            assert batch[0][0].shape[0] == batch_size
            break

        for batch in test_data_silo.batch_generator(DatasetName.VALID):
            assert batch[0][0].shape[0] == batch_size
            break
