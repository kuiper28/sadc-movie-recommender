import operator
from config import *

class DataSequence():

    def __init__(self, dataset):
        self.dataset = dataset
    def getUserSequenes(self):
        train_sequences = dict()
        users = []
        Items = set()
        history = dict()
        user = 0
        with open(TRAIN_FILE, "r") as f:
            for line in f:
                old_user = user
                user, item, rating, timestamp = line.strip().split("::")
                user = int(user)
                if user not in users and len(users) != 0:
                    sorted_history = sorted(history.items(), key=operator.itemgetter(1))
                    sorted_history = [i[0] for i in sorted_history]
                    train_sequences[old_user] = list(sorted_history[:])
                    history = {}
                    users.append(user)
                else:
                    if len(users) == 0:
                        users.append(int(user))
                    history[int(item)] = int(timestamp)
                    Items.add(int(item))

            if len(history) > 0:
                sorted_history = sorted(history.items(), key=operator.itemgetter(1))
                sorted_history = [i[0] for i in sorted_history]
                train_sequences[user] = list(sorted_history[:])

        test_sequences = dict()
        with open(TEST_FILE, "r") as f:
            for line in f:
                user, item, rating, timestamp = line.strip().split("::")
                user = int(user)
                item = int(item)
                if user not in test_sequences:
                    test_sequences[user] = set()
                test_sequences[user].add(item)

        return train_sequences, users, Items, test_sequences