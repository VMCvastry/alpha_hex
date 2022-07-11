from __future__ import annotations
from utils.logger import logging
from concurrent.futures import ThreadPoolExecutor

from utils.custom_dataset import CustomDataset
from mcts.mcst import MCTS
from game import Game
from variables import *
from net.trainer import Trainer, split_board
import torch
import threading


def rotate_left(board: list[list[int]]):
    return [list(reversed(row)) for row in zip(*board)]
    # return torch.cat([board[:,:,1:], board[:,:,:1]], dim=2)


def flip_board(board: list[list[int]]):
    return [list(reversed(row)) for row in board]


class PlayGame:
    def __init__(self, trainer):
        self.game: Game = Game()
        self.trainer = trainer
        self.states = []
        self.policies = []
        self.turn = []
        self.outcome = None

    def play(self):
        while True:
            player = MCTS(self.trainer, self.game.get_state(), self.game.player)
            move, policy = player.search()
            self.states.append(self.game.get_state())
            self.policies.append(policy)
            self.turn.append(self.game.player)
            self.game.set_mark(move)
            if self.game.winner is not None:
                logging.debug("{} wins!".format(self.game.winner))
                logging.debug(self.game)
                self.outcome = self.game.winner
                return self.outcome

    def get_tensors(self):
        states, priors = self.states, self.policies
        full_states, full_outcomes, full_priors = [], [], []
        full_turns = []
        for i in range(len(states)):
            cur_state, cur_policy = states[i], priors[i]
            full_turns.append(self.turn[i])
            full_states.append(cur_state)
            full_priors.append(cur_policy)
            # todo flip board
            # for j in range(3):
            #     cur_state = rotate_left(cur_state)
            #     cur_policy = rotate_left(cur_policy)
            #     full_states.append(cur_state)
            #     full_priors.append(cur_policy)
            #     full_turns.append(self.turn[i])
            # cur_state, cur_policy = flip_board(states[i]), flip_board(priors[i])
            # full_turns.append(self.turn[i])
            # full_states.append(cur_state)
            # full_priors.append(cur_policy)
            # for j in range(3):
            #     cur_state = rotate_left(cur_state)
            #     cur_policy = rotate_left(cur_policy)
            #     full_states.append(cur_state)
            #     full_priors.append(cur_policy)
            #     full_turns.append(self.turn[i])
        full_outcomes = [self.outcome * full_turns[i] for i in range(len(full_states))]

        outcomes = torch.Tensor(full_outcomes).unsqueeze(1)
        states = torch.tensor(
            [
                split_board(full_states[i], full_turns[i])
                for i in range(len(full_states))
            ],
            dtype=torch.float32,
        )
        priors = torch.tensor(full_priors)
        return states, outcomes, priors


def play_game_thread(trainer, data, n_game, lock):
    game = PlayGame(trainer)
    outcome = game.play()
    (states, outcomes, priors) = game.get_tensors()
    data.append((states, outcomes, priors))
    lock.acquire()
    n_game[0] += 1
    print(f"\rGame {n_game[0]}/{N_GAMES}", end=" ")
    lock.release()
    return outcome


def run_self_play(data_path, model_path):
    logging.warning("Starting self play")
    trainer = Trainer(model_name=model_path)
    data_set = None
    data = []
    n_game = [0]
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = executor.map(
            play_game_thread,
            [trainer] * N_GAMES,
            [data] * N_GAMES,
            [n_game] * N_GAMES,
            [lock] * N_GAMES,
        )
    for result in results:
        pass  # required otherwise the interpreter wont execute them
    for i in range(len(data)):
        if i == 0:
            data_set = CustomDataset(data[i][0], data[i][1], data[i][2])
        else:
            data_set.append(data[i][0], data[i][1], data[i][2])
    logging.warning("Finished self play")
    logging.info(f"dataset size: {data_set.__len__()}")
    data_set.store("./training_data", data_path)


if __name__ == "__main__":
    run_self_play("gen1", None)
