
import torch
import numpy as np
import util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from collections import deque

loss_deque = deque(maxlen=100)
train_loss = []

def trainer(opts, RNN, RNN_optim, criterion, data_loader):

    steps = 0
    for e in range(opts.epoch):

        for data, labels, lengths in data_loader:
            steps += 1

            data, labels, lengths = util.sort_batch(data, labels, lengths)

            RNN_optim.zero_grad()
            pred = RNN(data, lengths)
            loss = criterion(pred, labels.to(device))
            loss.backward()
            RNN_optim.step()

            loss_deque.append(loss.cpu().item())
            train_loss.append(np.mean(loss_deque))

            if steps % opts.print_every == 0:
                print('Epoch: {}, Steps: {}, Loss: {:.4},'.format(e, steps, loss.item()))

    util.raw_score_plotter(train_loss)
