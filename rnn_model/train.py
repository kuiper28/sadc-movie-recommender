import torch
import torch.nn as nn
import numpy as np
from model import MovieRecommender
from torch.autograd import Variable
from construct_dataset import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Hyper Parameters
num_layers = 1
hidden_size = 50
num_epochs = 10
learning_rate = 0.1
clip = 0.25
USE_CUDA = False
seed = 1234734614
torch.manual_seed(seed)


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


if USE_CUDA:
    torch.cuda.manual_seed(seed)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(num_items):

    correct = 0
    total = 0
    hidden = rnn.init_hidden()

    for user in test_sequences:
        if user not in train_sequences:
            continue
        sequence = train_sequences[user]
       
        input_variable = Variable(torch.LongTensor(sequence))
        output, hidden = rnn(input_variable, hidden)
        output = output[-1, :]
        hidden = repackage_hidden(hidden)
        target = test_sequences[user]

        topv, topi = output.data.topk(len(target))
        predicted = set(topi[:len(target)])

        correct += len(predicted.intersection(target))
        total += len(target)

    acc = correct/ float(total)
    print ("Test set accuracy", acc)
    return acc

data_sequence = DataSequence("MovieLens")
train_sequences, Users, Items, test_sequences = data_sequence.getUserSequenes()
# print "#TrainSequence, #Users, #Items", len(train_sequences), len(Users), max(Items)

rnn = MovieRecommender(max(Items) + 1, hidden_size, num_layers)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad,rnn.parameters()), lr=learning_rate)

accuracy = 0
epoch_loss = []
epoch_accuracy = []
train_accuracy = []
epochs = []
try:
    for epoch in range(1, num_epochs + 1):

        epochs.append(epoch)
        hidden = rnn.init_hidden()
        loss_total = 0
        acc = 0
        for i, sequence in enumerate(Users):

            sequence = train_sequences[sequence]

            input_variable = Variable(torch.LongTensor(sequence[:-1]))
            targets = sequence[1:]
            target_variable = Variable(torch.LongTensor(targets))

            # hidden = repackage_hidden(hidden)
            rnn.zero_grad()

            output = rnn(input_variable)
            loss = criterion(output, target_variable.contiguous().view(-1))
           
            val = (target_variable.data.view(-1).eq(torch.max(output, 1)[1].data).sum())
            acc += (val/float(len(output.data)))
            loss.backward()

            
            torch.nn.utils.clip_grad_norm(rnn.parameters(), clip)
            optimizer.step()

            loss_total += loss.data

        if epoch > 0:
            acc = acc/float(i+1)
            print("Total loss for epoch", epoch, loss_total)
            print("Train accuracy ", acc)
            epoch_loss.append(loss_total)
            train_accuracy.append(acc)

	    # sys.stdout.flush()
        curr_accuracy = evaluate(max(Items)+1)
        epoch_accuracy.append(curr_accuracy)
        if curr_accuracy > accuracy:
           accuracy = curr_accuracy
           with open('model.pt', 'wb')   as f:
               torch.save(rnn, f)


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

plt.close("all")
plt.plot(epoch_loss)
plt.ylabel("Total Loss")
plt.xlabel("Epochs")
plt.savefig('epoch_loss.png')
plt.clf()

plt.plot(epochs, epoch_accuracy, 'r', label='Test')
plt.plot(epochs, train_accuracy, 'b', label='Train')
plt.ylabel("Train & Test accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.savefig('epoch_acc.png')
plt.clf()

with open('model.pt', 'rb') as f:
    rnn = torch.load(f)

print("Best accuracy is ")
evaluate(max(Items)+ 1)