
import torch
import numpy as np
import util
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class test():
    def __init__(self, opts, RNN, RNN_optim, data_loader, checkpoint_path = './model/checkpoint_-1.pth'):

        self.opts = opts
        self.RNN = RNN
        self.RNN_optim = RNN_optim
        self.data_loader = data_loader
        self.checkpoint_path = checkpoint_path

    def tester(self):

        self.RNN.eval()

        num_correct = 0.
        num_comparison = 0.

        best_paths = []
        real_labels = []

        for data, labels, lengths in self.data_loader:
            data, labels, lengths = util.sort_batch(data, labels, lengths)

            path_score, best_path = self.RNN.get_tags(data.to(device), lengths.to(device))

            best_path = np.concatenate(best_path, 1)

            for i, lens in enumerate(lengths):
                best_paths.append(best_path[i][:lens])
                real_labels.append(labels[:, i][:lens])

        for pred, real in zip(best_paths, real_labels):
            num_correct += np.sum(real.numpy() == np.asarray(pred))
            num_comparison += len(real.numpy())

        print("Test Accuracy : ", num_correct / float(num_comparison))