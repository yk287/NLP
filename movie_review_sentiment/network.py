
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, opts, word_dict):
        super(RNN, self).__init__()

        # configuration options
        self.opts = opts

        # embedding layer
        self.embedding = nn.Embedding(len(word_dict.keys()), self.opts.embedding_dim)

        #the model
        self.rnn = nn.LSTM(input_size=self.opts.embedding_dim, hidden_size=self.opts.hidden_size, num_layers=self.opts.num_layers, dropout=self.opts.dropout, bidirectional=True)

        #linear layer
        self.linear = nn.Linear(self.opts.hidden_size * 2, self.opts.num_classes)
        self.dropout = nn.Dropout(self.opts.dropout)

        #softmax layer.
        self.softmax = nn.LogSoftmax()

    def init_hidden(self, batch_size):
        '''
        initialize
        :param batch_size:
        :return:
        '''

        if self.bidirectional:
            num_layers = self.opts.num_layers * 2
        else:
            num_layers = self.opts.num_layers
        return (autograd.Variable(torch.randn(num_layers, batch_size, self.opts.hidden_size).to(device)),
                autograd.Variable(torch.randn(num_layers, batch_size, self.opts.hidden_size).to(device)))

    def forward(self, batch, len):

        #self.hidden = self.init_hidden(batch.size(-1))

        #transforms the input into word embedding
        embeds = self.embedding(batch.to(device))

        packed_input = pack_padded_sequence(embeds, len)

        output, (ht, ct) = self.rnn(packed_input, None)

        #concat the last hidden layer of forward and backward LSTM
        ht = self.dropout(torch.cat((ht[-2, :, :], ht[-1, :, :]), dim=1))

        output = self.dropout(ht)
        output = self.linear(output)
        output = self.softmax(output)

        return output
