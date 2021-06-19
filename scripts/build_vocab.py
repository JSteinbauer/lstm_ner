import os
from collections import Counter
from pathlib import Path

# TODO: modify this depending on your needs (1 will work just fine)
# You might also want to be more clever about your vocab and intersect
# the GloVe vocab with your dataset vocab, etc. You figure it out ;)
from typing import Set


def get_words() -> Set[str]:
    # Get Counter of words on all the data, filter by min count, save
    def words(name):
        return os.path.join(DIRECTORY, f'{name}.words.txt')

    print('Build vocab words (may take a while)')
    counter_words = Counter()
    for n in ['train', 'testa', 'testb']:
        with Path(words(n)).open() as f:
            for line in f:
                counter_words.update(line.strip().split())

    vocab_words = {w for w, c in counter_words.items() if c >= MIN_COUNT}

    with open(os.path.join(DIRECTORY, 'vocab.words.txt'), 'w') as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(
        len(vocab_words), len(counter_words)))

    return vocab_words


def get_chars(vocab_words: Set[str]) -> None:
    """ Get all the characters from the vocab words """
    print('Build vocab chars')
    vocab_chars = set()
    for w in vocab_words:
        vocab_chars.update(w)

    with open(os.path.join(DIRECTORY, 'vocab.chars.txt'), 'w') as f:
        for c in sorted(list(vocab_chars)):
            f.write('{}\n'.format(c))
    print('- done. Found {} chars'.format(len(vocab_chars)))


def get_tags() -> None:
    """ Get all tags from the training set """

    def tags(name):
        return os.path.join(DIRECTORY, f'{name}.tags.txt')

    print('Build vocab tags (may take a while)')
    vocab_tags = set()
    with Path(tags('train')).open() as f:
        for line in f:
            vocab_tags.update(line.strip().split())

    with open(os.path.join(DIRECTORY, 'vocab.tags.txt'), 'w') as f:
        for t in sorted(list(vocab_tags)):
            f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_tags)))


def main() -> None:

    # 1. Words
    vocab_words = get_words()
    # 2. Get chars
    get_chars(vocab_words)
    # 3. Get tags
    get_tags()


if __name__ == '__main__':
    # Data directory
    DIRECTORY: str = 'data/conll-2003_preprocessed/'
    # Minimal number of word count to add it to vocabulary
    MIN_COUNT = 1

    main()

