
import torch
import numpy as np

import util
import confusionMatrix
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tester(opts, RNN, data_loader):

    #label to be fed into confusion matrix plot.

    #eval mode to stop dropout
    RNN.eval()

    #used for overall accuracy
    correct = 0
    total = 0

    pred_list = []
    labels_list = []
    y_label = np.arange(opts.num_classes)
    y_label = [str(e) for e in y_label]

    for data, label, lengths in data_loader:

        data, label, lengths = util.sort_batch(data, label, lengths)

        #run the data through RNN
        pred = RNN(data, lengths)

        #pick the argmax
        output = torch.max(pred, 1)[1]

        for output, label in zip(output, label):
            pred_list.append(output.cpu().item())
            labels_list.append(label.item())
            if output.cpu().item() == label.item():
                correct += 1
            total += 1

    confusionMatrix.plot_confusion_matrix(np.array(labels_list, dtype=np.int), np.array(pred_list, dtype=np.int), np.array(y_label), title="ConfusionMatrix")
    plt.show()

    print("Test Accuracy", correct / float(total))


