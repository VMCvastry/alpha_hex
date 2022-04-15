import logging

from game import Game
from mcts.mcst import MCTS
from trainer import Trainer
from collections import Counter
from variables import *

logging.basicConfig(level=logging.INFO)


def turn(game, move):
    game.set_mark(move)
    logging.info(game)
    if game.check_tic_tac_toe() is not None:
        logging.info("{} wins!".format(game.check_tic_tac_toe()))

    return game.check_tic_tac_toe()


def duel(trainer1: Trainer, trainer2: Trainer):
    game = Game()
    while 1:
        player = MCTS(trainer1, game.get_state(), game.player, exploration=1.4)
        move, _ = player.search()
        res = turn(game, move)
        if res is not None:
            return res
        player = MCTS(trainer2, game.get_state(), game.player, exploration=1.4)
        move, _ = player.search()
        res = turn(game, move)
        if res is not None:
            return res


def find_best(model_1, model_2):
    trainer1 = Trainer(model_name=model_1)
    trainer2 = Trainer(model_name=model_2)
    res = []
    for i in range(N_GAMES_DUEL):
        print(f"\rDuel {i}/{N_GAMES_DUEL}", end=" ")
        if i < N_GAMES_DUEL // 2:
            res.append(duel(trainer1, trainer2))
        else:
            res.append(-1 * duel(trainer2, trainer1))
    c = Counter(res)
    print(res)
    print(c)
    return c


if __name__ == "__main__":
    model_1 = "NET_2022-04-14_14-11-48"
    model_2 = "NET_2022-04-14_09-29-42"
    print(find_best(model_1, model_2))
