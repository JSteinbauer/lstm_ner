import os
import errno
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple


@dataclass
class InputExample:
    """A single training/test example for simple sequence classification.
    Args:
        guid: string? Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None


@dataclass
class InputFeatures:
    """ A single set of features of data. """

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


class NerProcessor(object):
    """ Processor for the CoNLL-2003 data set. """

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
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

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
        return cls.readfile(input_file)

    @staticmethod
    def readfile(filename: str) -> List[Tuple[List[str], List[str]]]:
        """ Read conll-2003 files and transforms them into a format that can be used for training """

        f = open(filename)
        sentence_label_list: List[Tuple[List[str], List[str]]] = []
        sentence: List[str] = []
        label: List[str] = []
        # Iterate over all lines
        for line in f:
            # If one of these criteria is met, finish sentences and append them to list
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    sentence_label_list.append((sentence, label))
                    sentence = []
                    label = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[-1][:-1])

        # Add last sentence to sentence_label_list list
        if len(sentence) > 0:
            sentence_label_list.append((sentence, label))
        return sentence_label_list


if __name__ == '__main__':
    DATA_DIR = "data/conll-2003/"

    OUT_DIR = "data/conll-2003_preprocessed/"
    Path(OUT_DIR).mkdir(exist_ok=True)

    max_seq_length = 128
    processor = NerProcessor()

    dataset_train = processor.get_train_examples(DATA_DIR)
    dataset_valid = processor.get_dev_examples(DATA_DIR)
    dataset_test = processor.get_test_examples(DATA_DIR)

    data_dict = {'train': dataset_train, 'testa': dataset_valid, 'testb': dataset_test}

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
            else:  # Something unexpected went wrong so re-raise the exception.
                raise
        else:  # No exception, so the file must have been created successfully.
            with open(wordsfile, 'w') as word_file, open(tagsfile, 'w') as tags_file:

                for data in data_set:
                    word_file.write(data.text_a + '\n')
                    tags_file.write(' '.join(data.label) + '\n')
