
import torch
import numpy as np

import util
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from collections import deque

def trainer(opts, RNN, RNN_optim, criterion, loader):

    last_100_loss = deque(maxlen=100)
    last_100_g_loss =[]

    iter_count = 0

    for epoch in range(opts.epoch):

        for param_group in RNN_optim.param_groups:
            param_group['lr'] = util.linear_LR(epoch, opts)
            print('Epoch: {}, D_LR: {:.4}'.format(epoch, param_group['lr']))

        for image, label in loader:

            '''Images'''
            image = image.view(-1, 28, 28)
            image = image.to(device)

            label = label.to(device)

            '''run the data through RNN'''
            output = RNN(image)
            loss = criterion(output, label)

            '''take a gradient step'''
            RNN_optim.zero_grad()
            loss.backward()
            RNN_optim.step()  # One step Descent into loss

            '''plot the loss'''
            last_100_loss.append(loss.item())
            last_100_g_loss.append(np.mean(last_100_loss))
            util.raw_score_plotter(last_100_g_loss)

            '''Train Generator'''
            iter_count += 1

            if iter_count % opts.print_every == 0:
                print('Epoch: {}, Iter: {}, Loss: {:.4},'.format(epoch, iter_count, loss.item()))


