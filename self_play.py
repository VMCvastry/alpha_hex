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
import threading

# logging.basicConfig(level=logging.INFO)


class PlayGame:
    def __init__(self, trainer):
        self.game = Game()
        self.trainer = trainer
        self.states = []
        self.policies = []
        self.outcome = None

    def play(self):
        while True:
            player = MCTS(self.trainer, self.game.get_state())
            move, policy = player.search()
            self.states.append(self.game.get_state())
            self.policies.append(policy)
            self.game.set_mark(move)
            if self.game.check_tic_tac_toe() is not None:
                logging.info("{} wins!".format(self.game.check_tic_tac_toe()))
                self.outcome = self.game.check_tic_tac_toe()
                return self.outcome

    def get_tensors(self):
        states, priors = self.states, self.policies
        outcomes = (
            torch.Tensor([self.outcome] * len(states))
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        states = torch.tensor(
            [split_board(state) for state in states], dtype=torch.float32
        )
        priors = torch.tensor(priors)
        return states, outcomes, priors


def play_game_thread(trainer, data, n_game, lock):
    game = PlayGame(trainer)
    outcome = game.play()
    (states, outcomes, priors) = game.get_tensors()
    data.append((states, outcomes, priors))
    lock.acquire()
    n_game += 1
    lock.release()
    return outcome


def run_self_play(data_path, model_path):
    logging.warning("Starting self play")
    trainer = Trainer(model_name=model_path)
    data_set = None
    data = []
    n_game = 0
    lock = threading.Lock()
    while True:
        if n_game > N_GAMES:
            break
        print(f"\rGame {n_game}/{N_GAMES}", end=" ")
        play_game_thread(trainer, data, n_game, lock)

    for i in range(len(data)):
        if i == 0:
            data_set = CustomDataset(data[i][0], data[i][1], data[i][2])
        else:
            data_set.append(data[i][0], data[i][1], data[i][2])
    logging.warning("Finished self play")
    print(data_set.__len__())
    data_set.store("./training_data", data_path)


if __name__ == "__main__":
    run_self_play("gen1", None)
