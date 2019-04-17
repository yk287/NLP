import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(([0.5]), ([0.5])),
    ])

#options
from options import options
options = options()
opts = options.parse()

#Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train= True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch_size, shuffle=True)

testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

from network import RNN
from train import trainer
from test import tester

'''RNN model'''
RNN = RNN(opts).to(device)

if opts.print_model:
    print(RNN)

'''Optimizers'''
import torch.optim as optim

RNN_optim = optim.Adam(RNN.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

'''Criterion'''
criterion = nn.CrossEntropyLoss()  # the target label is not one-hotted

'''run training'''
trainer(opts, RNN, RNN_optim, criterion, trainloader)

'''test'''
tester(opts, RNN, testloader)

