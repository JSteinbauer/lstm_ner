__author__ = "Jakob Steinbauer"

import os
import errno
from pathlib import Path
import numpy as np

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        
        
def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data


class NerProcessor(object):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
 
    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)
    
    
    
if __name__ == '__main__':
    DATA_DIR = "../CoNLL-2003/"
    #BERT_DIR = "data/multi_cased_L-12_H-768_A-12/"

    OUT_DIR = "../data/Data_Guillaume_Genthial/CoNLL_reformated/"
    Path(OUT_DIR).mkdir(exist_ok=True)

    max_seq_length = 128
    processor = NerProcessor()

    dataset_train = processor.get_train_examples(DATA_DIR)
    dataset_valid = processor.get_dev_examples(DATA_DIR)
    dataset_test = processor.get_test_examples(DATA_DIR)

    data_dict = {'train':dataset_train, 'testa': dataset_valid, 'testb':dataset_test}

    for name, data_set in data_dict.items():
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            wordsfile = os.open(os.path.join(OUT_DIR, name + ".words.txt"), flags)
            tagsfile = os.open(os.path.join(OUT_DIR, name + ".tags.txt"), flags)
        except OSError as e:
            if e.errno == errno.EEXIST:  # Failed as the file already exists.
                print("File " + OUT_DIR + name + ".words.txt" + " already exists!!!")
                print("File " + OUT_DIR + name + ".tags.txt" + " already exists!!!")
                pass
            else:  # Something unexpected went wrong so reraise the exception.
                raise
        else:  # No exception, so the file must have been created successfully.
            with open(wordsfile, 'w') as word_file, open(tagsfile, 'w') as tags_file:

                for data in data_set:
                    word_file.write(data.text_a + '\n')
                    tags_file.write(' '.join(data.label) + '\n')


    
