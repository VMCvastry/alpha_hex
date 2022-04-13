import logging

from game import Game
from mcts.mcst import MCTS
from trainer import Trainer
from collections import Counter
from variables import *

logging.basicConfig(level=logging.INFO)


def turn(game, move):
    game.set_mark(move)
    if game.check_tic_tac_toe() is not None:
        logging.info("{} wins!".format(game.check_tic_tac_toe()))
    return game.check_tic_tac_toe()


def duel(trainer1: Trainer, trainer2: Trainer):
    game = Game()
    while 1:
        player = MCTS(trainer1, game.get_state(), game.player)
        move, _ = player.search()
        res = turn(game, move)
        if res is not None:
            return res
        player = MCTS(trainer2, game.get_state(), game.player)
        move, _ = player.search()
        res = turn(game, move)
        if res is not None:
            return res


def find_best(trainer1: Trainer, trainer2: Trainer):
    res = []
    for _ in range(N_GAMES_DUEL // 2):
        res.append(duel(trainer1, trainer2))

    for _ in range(N_GAMES_DUEL // 2):
        res.append(-1 * duel(trainer2, trainer1))
    c = Counter(res)
    print(res)
    return c


if __name__ == "__main__":
    trainer1 = Trainer(model_name="NET_2022-04-12_21-27-23")
    trainer2 = Trainer(model_name="NET_2022-04-12_01-32-12")
    print(find_best(trainer1, trainer2))
