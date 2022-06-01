from __future__ import annotations

from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from custom_dataset import CustomDataset
from duel import find_best
from game import Game
from net import NET
from variables import *
from trainer import Trainer

from logger import logging

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
    logging.info(f"dataset len: {dataset.__len__()}")
    train_set, test_set = torch.utils.data.random_split(
        dataset,
        [len(dataset) - int(len(dataset) * TEST_RATIO), int(len(dataset) * TEST_RATIO)],
    )
    train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    trainer = Trainer(model_name=model_name)
    new_model_path = trainer.train(
        train,
        val_loader=[],
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
    )
    test_loss, val_loss, pol_loss = trainer.test(test)
    # trainer.plot_losses()
    logging.info(
        f"new model name: {new_model_path} trained on datasets {dataset_names}, test_loss: {test_loss}"
    )
    # trainer.plot_losses()
    return new_model_path


if __name__ == "__main__":
    model_name = "NEW_NET_2022-04-19_09-22-43"
    new_model_name = train_net(dataset_names=["gen23"], model_name=model_name)
    # res = find_best(new_model_name, model_name)
    # trainer = Trainer(model_name=model_name)
    # new_trainer = Trainer(model_name=new_model_name)
    # game = Game([[0, 0, -1], [1, 0, 0], [1, -2, 0]])
    # game = Game([[-1, 0, -1], [1, 0, 0], [-1, 1, 1]])
    # print(trainer.poll(game.get_state(), 1))
    # print(new_trainer.poll(game.get_state(), 1))
