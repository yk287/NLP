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
train_loader = util.create_dataset(data_loader.train_data, data_loader.letteridx, data_loader.labelidx, opts)
test_loader = util.create_dataset(data_loader.test_data, data_loader.letteridx, data_loader.labelidx, opts)

from network import RNN
from train import trainer
from test import tester

'''RNN model'''
RNN = RNN(opts, data_loader.letteridx).to(device)

if opts.print_model:
    print(RNN)

'''Optimizers'''
import torch.optim as optim

RNN_optim = optim.Adam(RNN.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

'''Criterion'''
criterion = nn.NLLLoss()

'''run training'''
trainer(opts, RNN, RNN_optim, criterion, train_loader)

'''test'''
tester(opts, RNN, test_loader)

