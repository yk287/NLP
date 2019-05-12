
import torch
import numpy as np
import util
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from collections import deque

class train():
    def __init__(self, opts, RNN, RNN_optim, data_loader, checkpoint_path = './model/checkpoint_-1.pth'):

        self.opts = opts
        self.RNN = RNN
        self.RNN_optim = RNN_optim
        self.data_loader = data_loader
        self.checkpoint_path = checkpoint_path

    def save_progress(self, epoch, loss):

        directory = './model/'
        filename = 'checkpoint_%s.pth' % epoch

        path = os.path.join('%s' % directory, '%s' % filename)

        torch.save({
            'epoch': epoch,
            'loss': loss,
            'RNN_state_dict': self.RNN.state_dict(),
            'RNN_optim_state_dict' : self.RNN_optim.state_dict(),
        }, path)

        print("Saving Training Progress")

    def load_progress(self,):

        # TODO get rid of hardcoding and come up with an automated way.

        checkpoint = torch.load(self.checkpoint_path)

        self.RNN.load_state_dict(checkpoint['RNN_state_dict'])
        self.RNN_optim.load_state_dict(checkpoint['RNN_optim_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("Training Loss From Last Session")
        return epoch, loss

    def trainer(self):

        steps = 0

        loss_deque = deque(maxlen=100)
        train_loss = []

        last_epoch = 0
        if self.opts.resume:
            last_epoch, loss = self.load_progress()

        for e in range(self.opts.epoch - last_epoch):

            '''Adaptive LR Change'''
            for param_group in self.RNN_optim.param_groups:
                param_group['lr'] = util.linear_LR(e, self.opts)
                print('epoch: {}, RNN_LR: {:.4}'.format(e, param_group['lr']))

            if self.opts.save_progress:
                '''Save the progress before start adjusting the LR'''
                if e == self.opts.const_epoch:
                    self.save_progress(self.opts.const_epoch, np.mean(loss_deque))

                if e % self.opts.save_every == 0:
                    self.save_progress(e, np.mean(loss_deque))

            for data, labels, lengths in self.data_loader:
                steps += 1

                data, labels, lengths = util.sort_batch(data, labels, lengths)

                self.RNN_optim.zero_grad()

                loss = self.RNN(data.to(device), labels.to(device), lengths.to(device))
                loss.backward()
                self.RNN_optim.step()

                loss_deque.append(loss.cpu().item())
                train_loss.append(np.mean(loss_deque))

                if steps % self.opts.print_every == 0:
                    print('Epoch: {}, Steps: {}, Loss: {:.4}'.format(e, steps, loss.item()))
                    util.raw_score_plotter(train_loss)

        if self.opts.save_progress:
            '''Save the progress before start adjusting the LR'''
            self.save_progress(-1, np.mean(loss_deque))

        util.raw_score_plotter(train_loss)
