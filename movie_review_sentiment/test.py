
import torch
import numpy as np

import util

import confusionMatrix
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class tester():
    def __init__(self,opts,  RNN, data_loader, checkpoint_path = './model/checkpoint_-1.pth'):

        self.opts = opts
        self.RNN = RNN
        self.data_loader = data_loader
        self.checkpoint_path = checkpoint_path

    def load_progress(self, ):

        checkpoint = torch.load(self.checkpoint_path)

        self.RNN.load_state_dict(checkpoint['RNN_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("Training Loss From Last Session")
        return epoch, loss

    def tester(self):

        _, _ = self.load_progress()
        confusion = torch.zeros(self.opts.num_classes, self.opts.num_classes)

        #label to be fed into confusion matrix plot.
        pred_list = []
        labels_list = []
        y_label = np.arange(self.opts.num_classes)
        y_label = [str(e) for e in y_label]

        #eval mode to stop dropout
        self.RNN.eval()

        #used for overall accuracy
        correct = 0
        total = 0

        for data, label, lengths in self.data_loader:

            data, label, lengths = util.sort_batch(data, label, lengths)

            #run the data through RNN
            pred = self.RNN(data, lengths)

            #pick the argmax
            output = torch.max(pred, 1)[1]

            for output, label in zip(output, label):
                pred_list.append(output.cpu().item())
                labels_list.append(label.item())
                if output.cpu().item() == label.item():
                    correct += 1
                total += 1

        confusionMatrix.plot_confusion_matrix(np.array(labels_list, dtype=np.int), np.array(pred_list, dtype=np.int),
                                              np.array(y_label), title="ConfusionMatrix")
        plt.show()

        print("Test Accuracy", correct / float(total))


