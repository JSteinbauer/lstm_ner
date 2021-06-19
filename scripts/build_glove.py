"""
Build an np.array from some glove file and some vocab file
"""
import os
import numpy as np


def main() -> None:
    # Load vocab
    with open(os.path.join(VOCAB_DIR, 'vocab.words.txt'), 'r') as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.zeros((size_vocab, 300))

    # Get relevant glove vectors
    found = 0
    print('Reading GloVe file (may take a while)')
    with open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'), 'r') as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print(f'- At line {line_idx}')
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print(f'- done. Found {found} vectors for {size_vocab} words')

    # Save np.array to file
    np.savez_compressed('glove.npz', embeddings=embeddings)


if __name__ == '__main__':
    VOCAB_DIR: str = 'data/conll-2003_preprocessed/'
    GLOVE_DIR: str = 'data/glove/'
    main()
