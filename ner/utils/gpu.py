import logging
import tensorflow as tf

log = logging.getLogger(__name__)


def setup_strategy(num_gpus: int = 0) -> tf.distribute.Strategy:
    """ Set up a distribute strategy for tensorflow.

    Typical use:
        with strategy.scope():
            keras_model.compile()

    Args:
        num_gpus: number of GPUs to use

    Returns:
        distribute strategy

    Raises:
        ValueError: if invalid number of GPUs is provided
    """
    if num_gpus == 0:
        strategy: tf.distribute.Strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
        devices_str: str = 'CPU'
    elif num_gpus == 1:
        strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
        devices_str = 'GPU'
    elif num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        devices_str = f'{strategy.num_replicas_in_sync} GPUs'
    else:
        raise ValueError(f'Invalid number of GPUs {num_gpus} for a strategy. Expected >= 0.')

    log.info(f'Using {strategy} on {devices_str} ')
    return strategy