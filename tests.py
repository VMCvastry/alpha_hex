from __future__ import annotations
from logger import logging
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


def turn(game, move):
    game.set_mark(move)
    logging.info(game)
    if game.check_tic_tac_toe() is not None:
        logging.info("{} wins!".format(game.check_tic_tac_toe()))


#
# model = NET(2, HIDDEN_FEATURES, RESNET_DEPTH, VALUE_HEAD_SIZE).to(device)
# loss_fn = nn.MSELoss(reduction="mean")
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
trainer = Trainer(model_name="NET_2022-04-12_01-32-12")
game = Game([[1, 0, 0], [1, -1, 0], [-1, 1, -1]])
game = Game([[0, 0, 0], [1, -1, 0], [0, 1, -1]])
# print(game)
# print(trainer.poll(game.get_state(), 1))
# # game = Game()
#
# player = MCTS(trainer, game.get_state())
# a, b = player.search()
# print(a)
# print(b)
# print(opt.poll([[1, None, None], [1, -1, None], [-1, 1, -1]]))
game = Game()
while 1:
    player = MCTS(trainer, game.get_state())
    a, b = player.search()
    print(a)
    print(b)
    turn(game, a)
    x, y = map(int, input(":").split(" "))
    move = Game.Move(x, y, -1)
    # move = random.choice(list(game.get_available_moves()))
    turn(game, move)
