from __future__ import annotations

import logging

from game import Game
from old_mcst.mcst import MCTS

logging.basicConfig(level=logging.INFO)


def turn(game, move):
    game.set_mark(move)
    logging.info(game)
    if game.check_tic_tac_toe() is not None:
        logging.info("{} wins!".format(game.check_tic_tac_toe()))


def test():
    # game = Game([[None, None, None], [1, -1, None], [None, 1, -1]])
    # game = Game([[1, None, None], [1, -1, None], [-1, 1, -1]])
    game = Game()
    while 1:
        tree = MCTS(game.get_state())
        move = tree.search()
        turn(game, move)
        x, y = map(int, input(":").split(" "))
        move = Game.Move(x, y, -1)
        # move = random.choice(list(game.get_available_moves()))
        turn(game, move)


if __name__ == "__main__":
    test()
