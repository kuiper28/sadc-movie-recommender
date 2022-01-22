from neural_colaborative_filtering.config import BATCH_SIZE, DATASET_PATH, DEVICE, EPOCH, LEARNING_RATE, MODEL_PATH, WEIGHT_DECAY
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import Dataset_REC
from model import NCRF

def train(model, optimizer, data_loader, criterion, device, log_interval=1000):
    model.train()
    total_loss = 0
    for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % log_interval == 0:
            print('    - loss:', total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            # print("YYYYYYYYYYYYYYYYY: ", y)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    return roc_auc_score(targets, predicts)


def main(dataset_path, epoch, learning_rate,
         batch_size, weight_decay, device):
    device = torch.device(device)

    dataset = Dataset_REC(dataset_path)
    
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    field_dims = dataset.field_dims
    model = NCRF(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.5,
                                            user_idx=dataset.user_idx,
                                            item_idx=dataset.item_idx)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    best_valid_auc = -100
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        valid_auc = test(model, valid_data_loader, device)
        if (valid_auc > best_valid_auc):
            torch.save(model.state_dict(), MODEL_PATH)
            best_valid_auc = valid_auc
            print("Model saved!!!")
        print('epoch:', epoch_i, 'validation: auc:', valid_auc)
    
    model_test = NCRF(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.5,
                                            user_idx=dataset.user_idx,
                                            item_idx=dataset.item_idx)
    model_test.load_state_dict(torch.load(MODEL_PATH))
    test_auc = test(model_test, test_data_loader, device)
    print('test auc:', test_auc)


if __name__ == '__main__':
    main(DATASET_PATH, EPOCH, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, DEVICE)