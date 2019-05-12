
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
from collections import Counter, defaultdict, OrderedDict, namedtuple

import numpy as np

from torch.utils.data import Dataset

Sentence = namedtuple("Sentence", "words tags")


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

        self.opts = opts

        self.train_data, self.train_label, self.test_data, self.test_label = self.train_test_split(opts.split)
        self.word_dict, self.label_dict = self.get_all_words(self.train_data), self.get_all_words(self.train_label)

        '''dict for words'''
        self.wordIdx = self.letter_to_index(self.word_dict, ['PAD', 'UNK'])
        '''dict for tags'''
        self.labelIdx = self.letter_to_index(self.label_dict, ['START', 'END', 'PAD'])

    def findFiles(self, path):
        return glob.glob(path)

    def get_all_words(self, corpus):
        '''
        Goes through the training data and creates a default dict with words and counts for the word
        :return:
        '''

        total_counts = Counter()

        for sequence in corpus:
            for word in sequence:
                total_counts[word] += 1

        return total_counts

    def get_rid_of_top_k(self):
        '''
        Deletes the top k most commonly occuring words
        :return:
        '''

        for word, _ in self.word_dict.most_common(self.opts.topk):
            self.word_dict.pop(word)

    def read_tags(self, filename):
        """Read a list of word tag classes"""
        with open(filename, 'r') as f:
            tags = f.read().split("\n")
        return frozenset(tags)

    def read_data(self, filename):
        """Read tagged sentence data"""
        with open(filename, 'r') as f:
            """New sentence starts with \n\n"""
            """Each word in a sentence is separated by \n"""
            sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
        return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t") for l in s[1:]]))) for s in sentence_lines if s[0]))

    def get_data(self, file_path):
        '''
        Given a directory, reads in the names of the file and the contents
        :param file_path: path where the directory for the files resides
        :return: category_lines, all_categories.
        '''

        category_lines = {}
        self.all_categories = []

        for filename in self.findFiles(file_path):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            file = open(filename)
            lines = list(map(lambda x:x[:-1], file.readlines()))
            file.close()
            category_lines[category] = lines

        return category_lines, self.all_categories

    def train_test_split(self, split):
        '''
        Given the data that's loaded above, returns a data that's been split into training and test set.

        :param split: what % of the data should be allocated to the training dataset
        :return: training, test which are list of tuples (Label, Name)
        '''

        data = self.read_data(self.opts.DataPath)
        tag = self.read_tags(self.opts.TagPath)

        sentence = []
        tags = []
        for items in data.items():
            sentence.append(items[1].words)
            tags.append(items[1].tags)

        p = np.random.permutation(len(data.items()))

        data = np.asarray(sentence)[p]
        tags = np.asarray(tags)[p]

        cutoff = np.int(len(data) * split)

        train_x, train_y, test_x, test_y  = data[:cutoff], tags[:cutoff], data[cutoff:], tags[cutoff:]

        return train_x, train_y, test_x, test_y


    def letter_to_index(self, sequences, label = None):
        '''
        Goes through all sequences word for word and creates an entry in the word dictionary
        :param pad:
        :param unk:
        :return:
        '''

        if self.opts.remove_topk:
            self.get_rid_of_top_k()

        wordToIdx = defaultdict(int)

        if label is not None:
            for lab in label:
                wordToIdx[lab] = len(wordToIdx)

        for word in sequences.keys():
            wordToIdx[word] = len(wordToIdx)

        return wordToIdx

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






