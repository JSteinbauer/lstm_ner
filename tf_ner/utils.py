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


def is_dir(dir_path: Union[str, LocalPath], exception: bool = False) -> bool:
    dir_exists = os.path.isdir(dir_path)

    if exception and not dir_exists:
        raise NotADirectoryError(dir_path)

    return dir_exists

def create_dir(dir_path: Union[str, LocalPath], exception: bool = True) -> None:
    """Creates a directory and its super paths.
    Succeeds even if the path already exists."""
    if not is_dir(dir_path):
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except OSError as e:
            if exception and e.errno != errno.EEXIST:
                raise

    # check that the dir_path now indeed exists as a directory
    is_dir(dir_path, exception=exception)


def is_file(file_path: Union[str, LocalPath], exception: bool = False) -> bool:
    file_exists = os.path.isfile(file_path)

    if exception and not file_exists:
        raise FileNotFoundError(file_path)

    return file_exists