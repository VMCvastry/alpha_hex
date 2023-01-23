from __future__ import annotations


from utils.logger import logging
from game import Game
from mcts.mcst import MCTS
from net.trainer import Trainer
from collections import Counter
from variables import *


def turn(game, move):
    game.set_mark(move)
    logging.debug(game)
    if game.winner is not None:
        logging.debug("{} wins!".format(game.winner))
    return game.winner


def duel(trainer1: Trainer, trainer2: Trainer):
    game = Game()
    while 1:
        player = MCTS(
            trainer1,
            game.get_state(),
            game.player,
            exploration=1.4,
            temperature=1,
            simulations_cap=1,
        )
        move, _ = player.search()
        res = turn(game, move)
        if res is not None:
            return res
        player = MCTS(
            trainer2,
            game.get_state(),
            game.player,
            exploration=1.4,
            temperature=1,
            simulations_cap=1,
        )
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
    # model_1 = "NET_2022-04-17_09-05-35_BEST"
    # model_1 = "NEW_NET_2022-04-19_09-22-43"
    # model_1 = "NEW_NET_2022-04-20_08-46-14"
    # model_1 = "NEW_NET_2022-04-21_09-32-57_BEST"
    #
    # # model_1 = "NEW_NET_2022-04-21_12-13-32"
    # # model_1 = "NEW_NET_2022-04-21_22-01-52"
    # # model_1 = "OLDEXP_FIXED_NET_2022-05-13_15-37-30"  # gen 58
    # # model_1 = "NEWEXP_FIXED_NET_2022-05-13_16-47-13"  # gen 53->61
    # model_1 = "FIXED_NET_2022-05-14_18-29-24"  # gen 53->61
    # model_2 = "GEN_133_2022-05-24_06-34-46"  # gen 53->61

    model_1 = "REBORN_NET_2022-06-01_12-11-28"
    # model_1 = "REBORN_NET_2022-06-01_12-35-11"

    model_1 = "REBORN_NET_2022-06-02_17-35-19"
    model_2 = "WINNER_NET_2022-06-03_18-16-50"

    model_1 = "R_3_HEX_NET_2022-12-27_06-37-56"
    model_1 = "R_3_HEX_NET_2023-01-03_20-03-18"
    model_2 = "R_3_HEX_NET_2023-01-08_16-31-31"

    print(find_best(model_1, model_2))
