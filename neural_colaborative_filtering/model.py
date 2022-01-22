import torch
import torch.nn.Module as Module
import torch.nn.Embedding as Embedding
import torch.nn.Linear as Linear
import torch.nn.BatchNorm1d as BatchNorm1d
import torch.nn.ReLU as ReLU
import torch.nn.Dropout as Dropout
import torch.nn.Sequential as Sequential

class NCRF(Module):

    def __init__(self, field_dims, user_idx, item_idx, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.user_idx = user_idx
        self.item_idx = item_idx
        self.embedding = Embedding(sum(field_dims), embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.layers_list = list()
        self.input_dim = 0
        for dim in mlp_dims:
            self.layers_list.append(Linear(self.embed_output_dim, dim))
            self.layers_list.append(BatchNorm1d(dim))
            self.layers_list.append(ReLU())
            self.layers_list.append(Dropout(dropout))
            self.input_dim = dim
        self.layers_list.append(Linear(self.input_dim, 1))
        self.sequential = Sequential(*self.layers_list)
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