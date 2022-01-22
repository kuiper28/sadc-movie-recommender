import torch
from torch.nn import Module, Embedding, Linear, BatchNorm1d, ReLU, Dropout, Sequential
import numpy as np

class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout):
        super().__init__()
        layers = list()
        for dim in embed_dims:
            # print(embed_dim)
            layers.append(Linear(input_dim, dim))
            layers.append(BatchNorm1d(dim))
            layers.append(ReLU())
            layers.append(Dropout(p=dropout))
            input_dim = dim
        self.mlp = Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class NCRF(Module):

    def __init__(self, field_dims, user_idx, item_idx, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.user_idx = user_idx
        self.item_idx = item_idx
        print("eeeeee", embed_dim)
        self.embedding = Embedding(sum(field_dims), embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.sequential = MLP(self.embed_output_dim, mlp_dims, dropout)
        self.fc = Linear(mlp_dims[-1] + embed_dim, 1)

    def forward(self, x):
        
        x = self.embedding(x)
        user_x = x[:, self.user_idx].squeeze(1)
        item_x = x[:, self.item_idx].squeeze(1)
        x = self.sequential(x.view(-1, self.embed_output_dim))
        gmf = user_x * item_x
        x = torch.cat([gmf, x], dim=1)
        x = self.fc(x).squeeze(1)
        return torch.sigmoid(x)