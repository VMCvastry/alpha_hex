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

logging.basicConfig(level=logging.INFO)


class PlayGame:
    def __init__(self, trainer):
        self.game = Game()
        self.trainer = trainer
        self.states = []
        self.policies = []

    def play(self):
        while True:
            player = MCTS(self.trainer, self.game.get_state())
            move, policy = player.search()
            self.states.append(self.game.get_state())
            self.policies.append(policy)
            self.game.set_mark(move)
            if self.game.check_tic_tac_toe() is not None:
                logging.info("{} wins!".format(self.game.check_tic_tac_toe()))
                return self.game.check_tic_tac_toe()


def run_self_play():
    trainer = Trainer()
    data = None
    for _ in range(N_GAMES):
        game = PlayGame(trainer)
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
