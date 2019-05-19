import os
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from torch.utils.data import Dataset, DataLoader
from data import SeqDataset

def vectorized_data(data, item2id):
    '''
    For each sequence in the data this function goes through char in the sequence and returns a list of indices of the char
    in the char dictionary.

    :param data:
    :param item2id:
    :return:
    '''
    return [[item2id[token] if token in item2id else item2id['UNK'] for token in seq] for seq in data]

def pad_data(data, length):
    '''
    Creates a 0 tensor and assigns the values of the indices calculated using vectorized_data for the right indicies
    Using 0 here becuase "PAD" has index of 0. The model will think 0 => padded

    :param data: Input data that's been vectorized using vectorized_data()
    :param length: length of each sequence in the data.
    :return:
    '''

    seq_tensor = torch.zeros((len(data), length.max())).long()

    for idx, (vec, veclen) in enumerate(zip(data, length)):
        seq_tensor[idx, :veclen] = torch.LongTensor(vec)
    return seq_tensor

def check_features(data):
    '''
    For each sequence in the data this functino goes through each word and checks if the word is capitalized.
    Also returns a vector that indicates whether the word is at the start of the sentence.
    :param data:
    :return:
    '''

    capitalized_seq = []
    start_seq = []
    mixed_alpha_Num_seq = []
    #only_number_seq = []


    for sequence in data:
        start = []
        capitalized = []
        mixed_alpha_Num = []
        #only_numbers = []

        for idx, word in enumerate(sequence):
            #Iterate through and append 1 if the word is the first one in the setence
            if idx == 0:
                start.append(1)
            else:
                start.append(0)

            # Iterate through and append 1 if the first letter in the word is capitalized
            if word[0].isupper():
                capitalized.append(1)
            else:
                capitalized.append(0)

            # true if the word is a mix of alphabet and nunbers
            if word.isalnum() and word.isalpha() == False and word.isdigit() == False:
                mixed_alpha_Num.append(1)
            else:
                mixed_alpha_Num.append(0)
            """
            if word.isdigit():
                only_numbers.append(1)
            else:
                only_numbers.append(0)
            """

        start_seq.append(start)
        capitalized_seq.append(capitalized)
        mixed_alpha_Num_seq.append(mixed_alpha_Num)
        #only_number_seq.append(only_numbers)

    return [start_seq, capitalized_seq, mixed_alpha_Num_seq]


def create_dataset(data, inputindx, targetindx, opts):
    '''
    Creates a dataset by
    1) Taking a raw sequence and vectorizng it
    2) Pad the sequence from 1) with index value that matches "PAD" token
    3) feed the padded sequence from 2), target label and length tensor into dataloader defined in data.py

    :param data: tensor that holds input data AND label
    :param inputindx: a dict that holds index for char's
    :param targetindx: a dict that holds index for the label
    :param opts: Holds options argument
    :return: A dataloader class that can be called to iterate over batches of data.
    '''

    if opts.use_features:
        features = check_features(data[0])
    vectorized_input = vectorized_data(data[0], inputindx)
    vectorized_label = vectorized_data(data[1], targetindx)

    input_len = torch.LongTensor([len(char) for char in vectorized_input])

    padded_input = pad_data(vectorized_input, input_len).unsqueeze(0)
    padded_label = pad_data(vectorized_label, input_len)

    if opts.use_features:
        for feature in features:
            padded_input = torch.cat([padded_input, pad_data(feature, input_len).unsqueeze(0)], dim=0)

    return DataLoader(SeqDataset(padded_input, padded_label, input_len), batch_size=opts.batch_size, shuffle=opts.shuffle, num_workers=opts.cpu_count)

def sort_batch(data, targets, lengths):
    '''
    Before the data gets fed into LSTM or GRU, the data needs to be sorted in descending order and then transposed.

    :param data: input data
    :param targets: target label to be predicted
    :param lengths: length of the input data.
    :return: Sorted_data, target label and lengths
    '''

    length, sort_idx = lengths.sort(0, descending=True)
    sorted_data = data[:, :][sort_idx]
    target_tensor = targets[sort_idx]

    return sorted_data.transpose(1,2).transpose(0,1), target_tensor.transpose(0, 1), length


def plotter(env_name, num_episodes, rewards_list, ylim):
    '''
    Used to plot the average over time
    :param env_name:
    :param num_episodes:
    :param rewards_list:
    :param ylim:
    :return:
    '''
    x = np.arange(0, num_episodes)
    y = np.asarray(rewards_list)
    plt.plot(x, y)
    plt.ylim(top=ylim + 10)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Avg Rewards Last 100 Episodes")
    plt.title("Rewards Over Time For %s" %env_name)
    plt.savefig("progress.png")
    plt.close()

def raw_score_plotter(scores):
    '''
    used to plot the raw score
    :param scores:
    :return:
    '''
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Train Loss')
    plt.xlabel('Number of Iterations')
    plt.title("Loss Over Time")
    plt.savefig("Train_Loss.png")
    plt.close()

def confusion_plot(matrix, y_category):
    '''
    A function that plots a confusion matrix
    :param matrix: Confusion matrix
    :param y_category: Names of categories.
    :return: NA
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + y_category, rotation=90)
    ax.set_yticklabels([''] + y_category)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

def linear_LR(epoch, opts):

    if epoch < opts.const_epoch:
        lr = opts.lr
    else:
        lr = np.linspace(opts.lr, 0, (opts.adaptive_epoch + 1))[epoch - opts.const_epoch]

    return lr

