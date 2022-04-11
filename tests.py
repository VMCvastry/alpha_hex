import logging

from custom_dataset import CustomDataset
from mcts.mcst import MCTS
from game import Game
from net import NET
from variables import *
from trainer import Trainer
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)

model = NET(2, HIDDEN_FEATURES, RESNET_DEPTH, VALUE_HEAD_SIZE).to(device)
loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
game = Game([[1, None, None], [1, -1, None], [-1, 1, -1]])
# game = Game([[None, None, None], [1, -1, None], [None, 1, -1]])
# game = Game()

player = MCTS(trainer, game.get_state())
print(player.search())
print(game)
# print(opt.poll([[1, None, None], [1, -1, None], [-1, 1, -1]]))
