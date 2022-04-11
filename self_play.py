import logging

from custom_dataset import CustomDataset
from mcts.mcst import MCTS
from game import Game
from net import NET
from variables import *
from trainer import Trainer, split_board
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# logging.basicConfig(level=logging.INFO)


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


def run_self_play(data_path, model_path):
    logging.warning("Starting self play")
    trainer = Trainer(model_name=model_path)
    data = None
    for n_game in range(N_GAMES):
        print(f"\rGame {n_game}/{N_GAMES}", end=" ")
        game = PlayGame(trainer)
        outcome = game.play()
        states, priors = game.states, game.policies
        outcomes = (
            torch.Tensor([outcome] * len(states)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        )
        states = torch.tensor(
            [split_board(state) for state in states], dtype=torch.float32
        )
        priors = torch.tensor(priors)
        if not data:
            data = CustomDataset(states, outcomes, priors)
        else:
            data.append(states, outcomes, priors)
    print(data.__len__())
    data.store("./training_data", data_path)


if __name__ == "__main__":
    run_self_play("gen1", None)
