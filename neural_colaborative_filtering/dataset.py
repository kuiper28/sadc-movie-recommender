import numpy as np
import pandas as pd
import torch.utils.data.Dataset as Datset


class Dataset_REC(Datset):

    def __init__(self, dataset_path, sep='::', engine='python', header=None):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]

        self.items = data[:, :2].astype(np.int) - 1
        # print("ITEMS: ", self.items.shape)
        self.targets = self.change_targets_values(data[:, 2]).astype(np.float32)
        # print("TARGETS: ", self.targets.shape)
        self.field_dims = np.max(self.items, axis=0) + 1

        self.user_idx = np.array((0,), dtype=np.long)
        self.item_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def change_targets_values(self, target):
        target[target < 4] = 0
        target[target >= 4] = 1
        return target
