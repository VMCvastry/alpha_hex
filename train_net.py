from __future__ import annotations

from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from custom_dataset import CustomDataset
from duel import find_best
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


def train_net(dataset_names: list[str], model_name):
    dataset: CustomDataset | None = None
    for i, name in enumerate(dataset_names):
        if i == 0:
            dataset = CustomDataset.load("./training_data", name)
        else:
            dataset.append_dataset(CustomDataset.load("./training_data", name))
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
    print(f"new model name: {new_model_path} trained on datasets {dataset_names}")
    # trainer.plot_losses()
    return new_model_path
    # print(trainer.evaluate(test, batch_size=1, n_features=2))


if __name__ == "__main__":
    model_name = "best_stupid_2022-04-14_09-29-42"
    new_model_name = train_net(dataset_names=["gen1"], model_name=model_name)
    res = find_best(new_model_name, model_name)
