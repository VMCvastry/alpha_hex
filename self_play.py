import logging

from custom_dataset import CustomDataset
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


class PlayGame:
    def __init__(self, opt):
        self.game = Game()
        self.opt = opt
        self.states = []
        self.policies = []

    def play(self):
        while True:
            player = MCTS(self.opt, self.game.get_state())
            move, policy = player.search()
            self.states.append(self.game.get_state())
            self.policies.append(policy)
            self.game.set_mark(move)
            if self.game.check_tic_tac_toe() is not None:
                logging.info("{} wins!".format(self.game.check_tic_tac_toe()))
                return self.game.check_tic_tac_toe()


def run_self_play():
    model = NET(2, hidden_features, RESNET_DEPTH, value_head_size).to(device)
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
    data = None
    for _ in range(N_GAMES):
        game = PlayGame(opt)
        outcome = game.play()
        states, priors = game.states, game.policies
        outcomes = torch.Tensor([outcome] * len(states))
        states = torch.Tensor(states)
        priors = torch.tensor(priors)
        if not data:
            data = CustomDataset(states, outcomes, priors)
        else:
            data.append(states, outcomes, priors)
    data.store("./training_data", "test")


if __name__ == "__main__":
    run_self_play()
