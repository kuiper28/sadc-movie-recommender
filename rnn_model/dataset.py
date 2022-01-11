import torch
import numpy as np

class ParseDataset(torch.utils.data.Dataset):
  def __init__(self, labels,  ids):
    self.labels = labels
    self.ids = ids
  
  def __len__(self):
    return len(self.labels)

  def get_batch_ratings(self, idx):
    return np.array(self.labels[idx])

  def get_batch_texts(self, idx):
    return self.ids[idx]

  def __getitem__(self, index):
    texts = self.get_batch_texts(index)
    ratings = self.get_batch_ratings(index)
    return texts, ratings