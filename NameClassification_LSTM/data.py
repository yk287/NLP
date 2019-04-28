
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

from collections import defaultdict
import numpy as np

from torch.utils.data import Dataset

class dataloader():
    def __init__(self, opts):
        '''
        A class that is used to read in and preprocess the dataset.
        Inspired by https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html most functions are
        from the tutorial

        The Class:
        1) Splits the data into train and test based on your input through opts
        2) creates a dict with character level index (includes UNK and PAD)
        3) creates a dict for labels
        4) converts all chars to Ascii

        :param opts: options argument that holds many arguments that can be parsed.
        '''

        self.path = opts.path
        self.opts = opts

        self.train_data, self.test_data = self.train_test_split(opts.split)
        self.letteridx = self.letter_to_index(self.all_letters, 'PAD', 'UNK')
        self.labelidx = self.letter_to_index(self.all_categories)

    def findFiles(self):
        return glob.glob(self.path)

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):

        self.all_letters = string.ascii_letters + " .,;'"

        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    # Build the category_lines dictionary, a list of names per language

    # Read a file and split into lines
    def readLines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicodeToAscii(line) for line in lines]

    def get_data(self):
        '''
        Given a directory, reads in the names of the file and the contents
        :param file_path: path where the directory for the files resides
        :return: category_lines, all_categories.
        '''

        category_lines = {}
        self.all_categories = []

        for filename in self.findFiles():
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.readLines(filename)
            category_lines[category] = lines

        return category_lines, self.all_categories

    def train_test_split(self, split):
        '''
        Given the data that's loaded above, returns a data that's been split into training and test set.

        :param split: what % of the data should be allocated to the training dataset
        :return: training, test which are list of tuples (Label, Name)
        '''


        data, categories = self.get_data()

        all_data = []

        for cat in categories:
            names = data[cat]

            for name in names:
                all_data.append((cat, name))

        np.random.shuffle(all_data)
        cutoff = np.int(len(all_data) * split)
        training, test = all_data[:cutoff], all_data[cutoff:]

        return training, test

    def letter_to_index(self, letter_set, pad=None, unk=None):

        letterToIdx = defaultdict(int)

        if unk is not None:
            letterToIdx[unk] = 1

        if pad is not None:
            letterToIdx[pad] = 0

        for letter in letter_set:
            letterToIdx[letter] = len(letterToIdx)

        return letterToIdx


class SeqDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, length_tensor):
        '''
        A nn.Dataset class that can be used to load data iteratively.

        :param data_tensor: A tensor that holds characters
        :param target_tensor: A tensor that holds target labels
        :param length_tensor: A tensor that holds length of data_tensor
        :param raw_data:
        '''

        assert data_tensor.size(0) == target_tensor.size(0) == length_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.length_tensor = length_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.length_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)






