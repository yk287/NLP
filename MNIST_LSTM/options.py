import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics

        # Training Options
        self.parser.add_argument('--epoch', type=int, nargs='?', default=5, help='total number of training episodes')
        self.parser.add_argument('--const_epoch', type=int, nargs='?', default=5, help='number of epochs where LR is constant')
        self.parser.add_argument('--adaptive_epoch', type=int, nargs='?', default=0, help='number of epochs where LR changes')

        self.parser.add_argument('--show_every', type=int, nargs='?', default=100, help='How often to show images')
        self.parser.add_argument('--print_every', type=int, nargs='?', default=100, help='How often to print scores')
        self.parser.add_argument('--cpu_count', type=int, nargs='?', default=2, help='number of cpu used for dataloading')

        self.parser.add_argument('--print_model', type=bool, nargs='?', default=True, help='Prints the model being used')

        self.parser.add_argument('--lr', type=int, nargs='?', default=0.0005, help='learning rate')
        self.parser.add_argument('--beta1', type=int, nargs='?', default=0.5, help='learning rate')
        self.parser.add_argument('--beta2', type=int, nargs='?', default=0.999, help='learning rate')

        self.parser.add_argument('--resume', type=bool, nargs='?', default=False, help='Resume Training')
        self.parser.add_argument('--save_progress', type=bool, nargs='?', default=True, help='save training progress')

        self.parser.add_argument('--num_classes', type=int, nargs='?', default=10, help='number of classes')

        #LSTM Options
        self.parser.add_argument('--batch_size', type=int, nargs='?', default=32, help='batch size to be used')
        self.parser.add_argument('--input_size', type=int, nargs='?', default=28, help='number of expected features')
        self.parser.add_argument('--hidden_size', type=int, nargs='?', default=32, help='number of features in hidden state')
        self.parser.add_argument('--num_layers', type=int, nargs='?', default=3, help='number of recurrent layers')
        self.parser.add_argument('--dropout', type=float, nargs='?', default=0.4, help='dropout rates for the hidden layers')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt



"""
options = options()
opts = options.parse()
batch = opts.batch
"""