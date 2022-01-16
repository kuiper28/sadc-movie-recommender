import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, nb_movies, movies_embedding_dim, hidden_size):
        super(RNNModel,self).__init__()
        self.nb_movies = nb_movies
        self.movies_embedding_dim = movies_embedding_dim
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(nb_movies, movies_embedding_dim)
        self.lstm = nn.LSTM(movies_embedding_dim, hidden_size, batch_first=False)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(hidden_size, nb_movies)
        self.activation = nn.ReLU()
    def forward(self, x):
        embds = self.activation(self.linear1(x))
        out, _ = self.lstm(embds)
        out = self.dropout(out)
        out = self.linear2(out)
        return out