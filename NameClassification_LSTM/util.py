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
    return [[item2id[token] if token in item2id else item2id['UNK'] for token in seq] for _, seq in data]

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

    vectorized_input = vectorized_data(data, inputindx)

    input_len = torch.LongTensor([len(char) for char in vectorized_input])
    paddend_input = pad_data(vectorized_input, input_len)

    label_tensor = torch.LongTensor([targetindx[label] for label, _ in data])

    return DataLoader(SeqDataset(paddend_input, label_tensor, input_len), batch_size=opts.batch_size, shuffle=opts.shuffle)

def sort_batch(data, targets, lengths):
    '''
    Before the data gets fed into LSTM or GRU, the data needs to be sorted in descending order and then transposed.

    :param data: input data
    :param targets: target label to be predicted
    :param lengths: length of the input data.
    :return: Sorted_data, target label and lengths
    '''

    length, sort_idx = lengths.sort(0, descending=True)
    sorted_data = data[sort_idx]
    target_tensor = targets[sort_idx]

    return sorted_data.transpose(0, 1), target_tensor, length


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
