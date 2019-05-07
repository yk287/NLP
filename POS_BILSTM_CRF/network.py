#some of the functions are from https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

import torch
import torch.nn as nn

import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, opts, word_dict, label_dict, output_size=None):
        super(RNN, self).__init__()

        # configuration options
        self.opts = opts
        # dict for Tags
        self.label_dict = label_dict

        # embedding layer
        self.embedding = nn.Embedding(len(word_dict.keys()), self.opts.embedding_dim)

        #the model
        self.rnn = nn.LSTM(input_size=self.opts.embedding_dim, hidden_size=self.opts.hidden_size, num_layers=self.opts.num_layers, dropout=self.opts.dropout, bidirectional=True)

        #linear layer
        self.output_size = output_size
        if output_size == None:
            self.output_size = self.opts.num_classes

        self.linear = nn.Linear(self.opts.hidden_size * 2, output_size)
        self.dropout = nn.Dropout(self.opts.dropout)

        #transition probability
        self.transition = nn.Parameter(torch.randn(output_size, output_size))

        #no transitions to 'START' label
        self.transition.data[label_dict['START'], :] = -100000.
        #no transitions from 'END' label
        self.transition.data[:, label_dict['END']] = -100000.
        #no transitions from 'PAD' label
        self.transition.data[:, label_dict['PAD']] = -100000.
        #no transition to 'PAD'
        self.transition.data[label_dict['PAD'], :] = -100000.
        #allow transition from 'END' to 'PAD'
        self.transition.data[label_dict['PAD'], label_dict['END']] = 0.
        #allow transition from 'PAD' to 'PAD'
        self.transition.data[label_dict['PAD'], label_dict['PAD']] = 0.

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

    def lstm_outputs(self, batch, len):
        '''
        Function used to run the sentence in batch through the bi-LSTM.
        :param batch: batch of sentences
        :param len: length of sentences
        :return: output that's been fed through the linear layer. Has dimention of [Length_of_sentence, Batch, tag_size]
        '''

        embeds = self.embedding(batch.to(device))
        packed_input = pack_padded_sequence(embeds, len)

        output, (ht, ct) = self.rnn(packed_input, None)

        output, _ = pad_packed_sequence(output)
        '''probabily needs mask'''

        output = self.linear(output)

        return output

    def argmax(self, vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
    def log_sum_exp(self, vec):
        '''
        Numerical stable way of log_sum
        :param vec:
        :return:
        '''
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def sentence_score(self, lstm_feats, sentence):
        '''
        given the transition matrix, gives the score for the actual tag sequence
        :param sentence:
        :return:
        '''

        sentence_score = torch.zeros(1).to(device)

        tags = torch.cat([torch.tensor([self.label_dict['START']], dtype=torch.long).to(device), sentence.squeeze(1)])

        lstm_feats = lstm_feats.squeeze(1)
        for i, feat in enumerate(lstm_feats):
            sentence_score = sentence_score + self.transition[tags[i+1], tags[i]] + feat[tags[i+1]]

        '''add the score for the end tag'''
        sentence_score = sentence_score + self.transition[self.label_dict['END'], tags[-1]]

        return sentence_score

    def crf_forward(self, inputs):
        '''
        forward part of forward-backward algo to calculate the Z(X)
        quite helpful : https://web.stanford.edu/~jurafsky/slp3/A.pdf
        :param input:
        :return:
        '''

        #variable for forward algorithm size (batch, tag_size)
        init_alphas = torch.full((1, self.output_size), -100000.)

        init_alphas[0][self.label_dict['START']] = 0.

        forward_var = init_alphas
        inputs = inputs.squeeze(1)
        for input in inputs:
            alpha_at_t = []
            for next_tag in range(self.output_size):
                emission_score = input[next_tag].view(1, -1).expand(1, self.output_size)
                transition_score = self.transition[next_tag].view(1, -1)
                next_tag_score = forward_var + transition_score.cpu() + emission_score.cpu()
                alpha_at_t.append(self.log_sum_exp(next_tag_score).view(1))
            forward_var = torch.cat(alpha_at_t).view(1, -1)
        #add end state
        end_var = forward_var + self.transition[self.label_dict['END']].cpu()
        alpha = self.log_sum_exp(end_var)

        return alpha

    def viterbi_algo(self, lstm_feats):
        '''
        Viterbi Algorithm to get the path of sequence that has the highest score.
        :param lstm_feats:
        :return:
        '''

        backpointers = []

        '''initialize the viterbi algo with starting label START'''
        viterbi_vars = torch.full((1, self.output_size), -100000.)
        viterbi_vars[0][self.label_dict['START']] = 0
        forward = viterbi_vars.to(device)

        for feat in lstm_feats:

            backpoint_t = []
            viterbi_var_t = []

            for next_tag in range(self.output_size):

                next_tag_var = forward + self.transition[next_tag]
                best_tag = self.argmax(next_tag_var)
                backpoint_t.append(best_tag)
                viterbi_var_t.append(next_tag_var[0][best_tag].view(1))

            forward = (torch.cat(viterbi_var_t) + feat).view(1, -1)
            backpointers.append(backpoint_t)


        end_var = forward + self.transition[self.label_dict['END']]
        best_tag = self.argmax(end_var)
        path_score = end_var[0][best_tag]

        best_path = [best_tag]

        for bptrs_t in reversed(backpointers):
            best_tag = bptrs_t[best_tag]
            best_path.append(best_tag)

        start = best_path.pop()

        assert  start == self.label_dict['START']
        best_path.reverse()
        return path_score, best_path

    def get_tags(self, batch, len):
        '''
        Used during test time to get the predicted tags for a sentence.
        :param batch: tensor of sequence
        :param len: tensor of len of the sequence
        :return:
        '''

        lstm_features = self.lstm_outputs(batch, len)
        path_score, best_path = self.viterbi_algo(lstm_features)

        return path_score, best_path

    def forward(self, batch, tags, len):
        '''
        feeds the input sentence into bi-lstm and feeds the resulting output to the CRF which returns the Z(x)
        tags_score returns the score for the real tags.
        :param batch: batch of sentence to be used
        :param tags: batch of tags to be used
        :param len: batch of len of sentences
        :return: NLLLoss for the sentence
        '''

        '''get the LSTM outputs'''
        lstm_features = self.lstm_outputs(batch, len)
        '''get the Z(x) by using forward algorithm'''
        forward_algo_score = self.crf_forward(lstm_features)
        '''get the scores for the real tags'''
        tags_score = self.sentence_score(lstm_features, tags)

        return (forward_algo_score.to(device) - tags_score.to(device)) / len.float()
