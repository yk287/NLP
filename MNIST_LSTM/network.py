
import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, opts):
        super(RNN, self).__init__()

        # configuration options
        self.opts = opts

        self.rnn = nn.LSTM(input_size=self.opts.input_size, hidden_size=self.opts.hidden_size, num_layers=self.opts.num_layers, batch_first=True, dropout= self.opts.dropout)
        self.linear = nn.Linear(self.opts.hidden_size, self.opts.num_classes)

    def init_hidden(self):
        #probabily not necessary because of https://github.com/pytorch/pytorch/issues/434

        hidden = Variable(next(self.parameters()).data.new(self.opts.batch_size, self.opts.hidden_size), requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(self.opts.batch_size, self.opts.hidden_size), requires_grad=False)

        return hidden.zero_(), cell.zero_()

    def forward(self, x):

        '''If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.'''
        '''https://github.com/pytorch/pytorch/issues/434'''

        out, (h_n, h_c) = self.rnn(x, None)
        out = self.linear(out[:, -1, :])

        return out

