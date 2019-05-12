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
        self.word_dict = word_dict

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

        max_score = torch.max(vec, dim=1)[0]
        max_score_broadcast = max_score.unsqueeze(1).expand(vec.shape)
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))

    def sentence_score(self, lstm_feat, sentence, masks):
        '''
        given the transition matrix, gives the score for the actual tag sequence
        :param sentence:
        :return:
        '''

        batch_size = lstm_feat.shape[1]

        '''initialize the viterbi algo with starting label START'''
        score = torch.zeros(batch_size).to(device)

        start = torch.full((1, batch_size), self.label_dict['START']).long().to(device)
        tags = torch.cat([start, sentence], dim=0)
        mask = masks.unsqueeze(2)

        for i in range(lstm_feat.shape[0]):
            # get the mask values
            mask_t = mask[i, :].squeeze(1)
            score_t = score + self.transition[tags[i + 1], tags[i]] + lstm_feat[i, :, tags[i + 1]].diag()
            score = score * (1 - mask_t) + (score_t) * (mask_t)
        # append the score for the 'END' Tag
        score = score + self.transition[self.label_dict['END'], tags[-1]]

        return score

    def crf_forward(self, lstm_feat, masks):
        '''
        forward part of forward-backward algo to calculate the Z(X)
        quite helpful : https://web.stanford.edu/~jurafsky/slp3/A.pdf
        :param input:
        :param mask: tensor of masks
        :return:
        '''

        batch_size = lstm_feat.shape[1]

        '''initialize the viterbi algo with starting label START'''
        forward_var = torch.full((batch_size, self.output_size), -100000.).to(device)
        forward_var[:, self.label_dict['START']] = 0

        score = forward_var.to(device)
        mask = masks.unsqueeze(2)
        transition = self.transition.unsqueeze(0)
        for t in range(lstm_feat.shape[0]):
            #get the mask values
            mask_t = mask[t, :].to(device)
            #score is the previous score + transition socre + emission score
            score_t = score.unsqueeze(1) + transition + torch.transpose(lstm_feat[t, :, :].unsqueeze(1), 1, 2)
            #update the score
            score = score * (1 - mask_t) + torch.logsumexp(score_t, dim=2) * mask_t
        #append the score for the 'END' Tag
        score = torch.logsumexp(score + self.transition[self.label_dict['END']], dim=1)

        return score


    def viterbi_algo(self, lstm_feats, masks):
        '''
        Viterbi Algorithm to get the path of sequence that has the highest score.
        :param lstm_feats:
        :return:
        '''
        batch_size = lstm_feats.shape[1]
        '''initialize the viterbi algo with starting label START'''
        viterbi_vars = torch.full((batch_size, self.output_size), -100000.).to(device)
        viterbi_vars[:, self.label_dict['START']] = 0

        score = viterbi_vars.to(device)
        mask = masks.unsqueeze(2)

        back_pointer = []
        transition = self.transition.unsqueeze(0)
        for t in range(lstm_feats.shape[0]):
            mask_t = mask[t, :].to(device)
            score_t = score.unsqueeze(1) + transition
            score_t, best_prev_tag = torch.max(score_t, dim=2)
            back_pointer.append(best_prev_tag)
            score_t += lstm_feats[t, :, :]

            score = score * (1 - mask_t) + score_t * mask_t

        score += self.transition[self.label_dict['END']]
        final_score, best_tag = torch.max(score, dim=1)

        viterbi_path = [best_tag.cpu().unsqueeze(1).numpy()]
        tag = best_tag.cpu().unsqueeze(1)
        for t in reversed(back_pointer):
            step = t.cpu()
            tag = torch.t(step[:, tag].diagonal())
            viterbi_path.append(tag.numpy())

        viterbi_path.pop()
        viterbi_path.reverse()

        return final_score, viterbi_path

    def get_tags(self, batch, len):
        '''
        Used during test time to get the predicted tags for a sentence.
        :param batch: tensor of sequence
        :param len: tensor of len of the sequence
        :return:
        '''

        mask = (batch > self.word_dict['PAD']).float()
        lstm_features = self.lstm_outputs(batch, len)
        path_score, best_path = self.viterbi_algo(lstm_features, mask)

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

        '''create mask'''
        #TODO truncate mask to the size of lstm_features output.
        mask = (batch > self.word_dict['PAD']).float()

        '''get the LSTM outputs'''

        lstm_features = self.lstm_outputs(batch, len)
        '''get the Z(x) by using forward algorithm'''
        forward_algo_score = self.crf_forward(lstm_features, mask)
        '''get the scores for the real tags'''
        tags_score = self.sentence_score(lstm_features, tags, mask)

        return torch.sum((forward_algo_score - tags_score) / len.float())
