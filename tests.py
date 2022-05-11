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


def turn(game, move):
    game.set_mark(move)
    logging.info(game)
    if game.check_tic_tac_toe() is not None:
        logging.info("{} wins!".format(game.check_tic_tac_toe()))


#
# model = NET(2, HIDDEN_FEATURES, RESNET_DEPTH, VALUE_HEAD_SIZE).to(device)
# loss_fn = nn.MSELoss(reduction="mean")
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def test(trainer, game, player_turn):
    print(trainer.poll(game.get_state(), player_turn))

    player = MCTS(trainer, game.get_state(), player_turn)
    a, b = player.search()
    print(a)
    print(b)


# trainer = Trainer(model_name="best_stupid_2022-04-14_09-29-42")
# trainer = Trainer(model_name="NET_2022-04-17_09-05-35_BEST")
# new_trainer = Trainer(model_name="NEW_NET_2022-04-19_09-22-43")
game = Game([[1, 0, 0], [1, -1, 0], [-1, 1, -1]])  # test mtcs 1
game = Game([[0, 1, 1], [-1, 0, 0], [0, 0, 0]])
# game = Game([[1, 0, 0], [-1, 1, 0], [-1, -1, 0]])
# game = Game([[0, 0, 1], [0, -1, -1], [0, 0, 1]])  # test if value is 0 and other neg 1
# game = Game([[-1, 1, 1], [-1, 0, 0], [1, 0, 0]])
# game = Game([[-1, 0, -1], [1, 0, 0], [-1, 1, 1]])
player_turn = 1
print(game)
print("player turn: {}".format(player_turn))

# test(Trainer(model_name="NET_2022-04-17_09-05-35_BEST"), game, player_turn)
test(Trainer(model_name="NEW_NET_2022-04-21_09-32-57_BEST"), game, player_turn)
# test(Trainer(model_name="NEW_NET_2022-04-21_12-13-32"), game, player_turn)
# test(Trainer(model_name="NEW_NET_2022-04-21_22-01-52"), game, player_turn)
# test(Trainer(model_name="NEW_NET_2022-04-26_13-40-49"), game, player_turn)
# print(opt.poll([[1, None, None], [1, -1, None], [-1, 1, -1]]))
# game = Game(player=-1)
# while 1:
#     x, y = map(int, input(":").split(" "))
#     move = Game.Move(x, y, -1)
#     # move = random.choice(list(game.get_available_moves()))
#     turn(game, move)
#
#     player = MCTS(new_trainer, game.get_state(), 1)
#     a, b = player.search()
#     print(a)
#     print(b)
#     turn(game, a)
