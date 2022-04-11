from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from custom_dataset import CustomDataset
from net import NET
from variables import *
from trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train = torch.load("data/train_dataset.pt")
# val = torch.load("data/val_dataset.pt")
# test = torch.load("data/test_dataset.pt")
# train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
# val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
# test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
# test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
def train_net(data_path, model_name):
    dataset = CustomDataset.load("./training_data", data_path)
    print(dataset.__len__())
    train_set, test_set = torch.utils.data.random_split(
        dataset, [len(dataset) - TEST_LEN, TEST_LEN]
    )
    train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    trainer = Trainer(model_name=model_name)
    new_model_path = trainer.train(
        train,
        val_loader=[],
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        n_features=2,
    )
    print("new model path: ", new_model_path)
    return new_model_path
    # trainer.plot_losses()
    # print(trainer.evaluate(test, batch_size=1, n_features=2))


if __name__ == "__main__":
    train_net(data_path="gen1", model_name=None)
