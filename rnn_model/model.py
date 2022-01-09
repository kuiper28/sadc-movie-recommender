import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils
import numpy as np

class MovieRecommender(nn.Module):
    def __init__(self, num_items, hidden_size, number_of_layers):

        super(MovieRecommender, self).__init__()
        self.hidden_size = hidden_size
        self.number_of_layers = number_of_layers
        self.num_items = num_items

        self.embedding = nn.Embedding(num_items, hidden_size)
        self.lstm = nn.GRU(hidden_size, hidden_size, number_of_layers, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, num_items)
        self.init_weights()

    def forward(self, x, hidden):

        seq_len = len(x)
        
        embedings = self.embedding(x).view(seq_len, 1, -1)
        out, hidden = self.lstm(embedings, hidden)
        out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        out = self.linear(out)
        return out, hidden

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.number_of_layers*2, 1, self.hidden_size))
        hidden.cuda()
        return hidden