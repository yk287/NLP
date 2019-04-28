import torch
import torch.nn as nn

import data
import util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#options
from options import options
options = options()
opts = options.parse()

#data loader
data_loader = data.dataloader(opts)

train_loader = util.create_dataset([data_loader.train_data, data_loader.train_label], data_loader.wordIdx, data_loader.labelIdx, opts)

from network import RNN
from train import train
from test import tester

'''RNN model'''
RNN = RNN(opts, data_loader.wordIdx).to(device)

if opts.print_model:
    print(RNN)

'''Optimizers'''
import torch.optim as optim

RNN_optim = optim.Adam(RNN.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

'''Criterion'''
criterion = nn.NLLLoss()

'''run training'''
trainer = train(opts, RNN, RNN_optim, criterion, train_loader)
trainer.trainer()

'''test'''
test_loader = util.create_dataset([data_loader.test_data, data_loader.test_label], data_loader.wordIdx, data_loader.labelIdx, opts)
tester = tester(opts, RNN, test_loader)
tester.tester()

