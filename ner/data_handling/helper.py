from typing import List

import tensorflow as tf

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

    # Convert lists to tensors
    words_tensor = tf.convert_to_tensor([words], tf.string)
    nwords_tensor = tf.convert_to_tensor([nwords], tf.int32)
    tags_tensor = tf.convert_to_tensor([['O'] * nwords], tf.string)

    # Return directly if no character data is required
    if not use_chars:
        return [words_tensor, nwords_tensor, tags_tensor]

    # Prepare character data
    nchars = [len(word) for word in words]
    max_num_chars = max(nchars)
    chars = [[c for c in word] + ['<pad>'] * (max_num_chars - len(word)) for word in words]

    # Convert lists to tensors
    chars_tensor = tf.convert_to_tensor([chars], tf.string)
    nchars_tensor = tf.convert_to_tensor([nchars], tf.int32)
    return [words_tensor, nwords_tensor, chars_tensor, nchars_tensor, tags_tensor]
