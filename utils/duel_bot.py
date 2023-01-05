from __future__ import annotations

from mopyhex.gtpinterface import gtpinterface
from mopyhex.mctsagent import mctsagent
from utils.logger import logging
from mcts.mcst import MCTS
from game import Game, discard_rows_for_print
from net.trainer import Trainer
import torch

from variables import HEX_GRID_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def turn(game, move):
    game.set_mark(move)
    logging.info(game)
    if game.winner is not None:
        logging.info("{} wins!".format(game.winner))
        return game.winner


def duel(trainer, time_limit):
    player_turn = 1
    game = Game(player=player_turn)
    agent = mctsagent()
    interface = gtpinterface(agent, time_limit)
    interface.send_command(f"size {HEX_GRID_SIZE}")
    while 1:
        p, v = trainer.poll(game.get_state(), player_turn)
        [
            print(list(map("{:.2f}".format, x)))
            for x in discard_rows_for_print(p.tolist())
        ]
        # logging.info(discard_rows_for_print(p.tolist()))
        logging.info(v)
        player = MCTS(
            new_trainer,
            game.get_state(),
            1,
            exploration=1.4,
            temperature=0,
            simulations_cap=1,
        )
        a, b = player.search()
        logging.info(a)
        logging.info(discard_rows_for_print(b))
        real_x, real_y = a.x // 3, (a.y - (a.x // 3)) // 2
        logging.info(f"{real_x} {real_y}")
        interface.send_command(
            f"play {'w' if game.player==-1 else 'b'} {['A','B','C','D'][real_y]}{real_x+1}"
        )
        if turn(game, a) is not None:
            return game.winner

        res, move = interface.send_command(f"genmove {'w' if game.player==-1 else 'b'}")
        if not res:
            logging.info("Game over")

        y, x = move
        logging.info(f"bot move = {x} {y}")
        move = Game.Move(x * 3, y * 2 + x, -1)
        # move = random.choice(list(game.get_available_moves()))
        if turn(game, move) is not None:
            return turn(game, move)
        print(interface.send_command("showboard")[1])


if __name__ == "__main__":
    new_trainer = Trainer(model_name="R_3_HEX_NET_2022-12-27_06-37-56")

    result = 0
    for i in range(10):
        logging.info(f"Game {i}")
        result += duel(new_trainer, 0.05)
        logging.info(f"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\n\n{result}")
    logging.info(f"Result: {result}")
