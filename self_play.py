import logging

from mcts.mcst import MCTS
from game import Game
from net import NET
from variables import *
from trainer import Optimization
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)

model = NET(2, hidden_features, RESNET_DEPTH, value_head_size).to(device)
loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
game = Game([[1, None, None], [1, -1, None], [-1, 1, -1]])
# game = Game([[None, None, None], [1, -1, None], [None, 1, -1]])
# game = Game()
player = MCTS(opt, game.get_state())
print(player.search())
print(game)
# print(opt.poll([[1, None, None], [1, -1, None], [-1, 1, -1]]))


class PlayGame:
    def __init__(self, opt):
        self.game = Game()
        self.opt = opt
        self.states = []
        self.policies = []

    def play(self):
        while True:
            player = MCTS(opt, game.get_state())
            move, policy = player.search()
            self.states.append(game.get_state())
            self.policies.append(policy)
            game.set_mark(move)
            if game.check_tic_tac_toe() is not None:
                logging.info("{} wins!".format(game.check_tic_tac_toe()))
                return game.check_tic_tac_toe()


model = NET(2, hidden_features, RESNET_DEPTH, value_head_size).to(device)
loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
for _ in range(N_GAMES):
    game = PlayGame(opt)
    outcome = game.play()
    states, priors = game.states, game.policies
    outcomes = torch.Tensor([outcome] * len(states))
    states = torch.Tensor(states)
    priors = torch.tensor(priors)
