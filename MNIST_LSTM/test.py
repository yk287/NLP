
import torch
import numpy as np

import util
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tester(opts, RNN, loader):

    #Confusion matrix to be used

    confusion = torch.zeros(opts.num_classes, opts.num_classes)
    y_label = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    RNN.eval()
    for image, label in loader:

        '''Images'''
        image = image.view(-1, 28, 28)
        image = image.to(device)

        '''run the data through RNN'''
        output = RNN(image)
        output = torch.argmax(output)

        # Confusion matrix
        confusion[output.cpu().item()][label.item()] += 1

    for i in range(opts.num_classes):
        confusion[i] = confusion[i] / confusion[i].sum()

    util.confusion_plot(confusion, list(y_label))


