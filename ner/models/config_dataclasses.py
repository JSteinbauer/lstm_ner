from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class NerLstmConfig:
    # Model specific config
    embedding_dim: int
    char_embedding_dim: int
    lstm_size: int
    dropout: float
    num_oov_buckets: int
    buffer: int
    convolution_filters: int
    kernel_size: int

    # Training specific config
    learning_rate: float
    epochs: int
    batch_size: int

    # System config
    num_gpus: int

    @classmethod
    def default_config(cls) -> 'NerLstmConfig':
        return cls(
            embedding_dim=300,
            char_embedding_dim=100,
            lstm_size=100,
            dropout=0.5,
            num_oov_buckets=1,
            buffer=15000,
            convolution_filters=50,
            kernel_size=3,
            learning_rate=1e-3,
            epochs=20,
            batch_size=100,
            num_gpus=0,
        )
