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
        self.lstm = nn.LSTM(num_items, self.hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, num_items)
        self.init_weights()

    def forward(self, x,):
        
        embedings = self.embedding(x)
        out, _ = self.lstm(embedings)
        # out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        out = self.linear(out)
        return out

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.number_of_layers*2, 1, self.hidden_size))
        hidden.cuda()
        return hidden