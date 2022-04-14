from __future__ import annotations

import logging
import random
import numpy as np
from game import Game
from variables import *


def get_move_prior(priors, move: Game.Move):
    # print(priors)
    # print(priors[move.x][move.y])
    return float(priors[move.x][move.y])


class Aux_MCTS:
    # exploration_parameter = np.sqrt(2)

    class Node:
        def __init__(
            self,
            *args,
            state,
            prior,
            parent: Aux_MCTS.Node | None,
            move: Game.Move | None,
        ):
            self.state = state
            self.value = 0
            self.visits = 0
            self.prior = prior
            self.subs: set[Aux_MCTS.Node] = set()
            self.parent: Aux_MCTS.Node = parent
            self.move = move

        def get_mean_value(self):
            return self.value / (self.visits + 0.00001)

        def get_next_player(self):
            if not self.move:
                return 1
            else:
                return -1 * self.move.mark

        def layer_visits(self):
            subs = self.parent.subs
            return sum((s.visits for s in subs))

        def interest(self, exploration):
            return self.get_mean_value() + exploration * (
                self.prior * np.sqrt(self.layer_visits()) / (self.visits + 1)
            )

        def __repr__(self):
            return f"{self.move},value {self.value}, visits={round(self.visits)}"

    @staticmethod
    def flip_state(state, player):
        return [[player * s for s in row] for row in state]

    @staticmethod
    def pick_best_move(node: Aux_MCTS.Node, temperature):
        logging.info([str(n) for n in node.subs])
        # denominator = node.layer_visits() ** (1 / temperature)
        best_node = max(node.subs, key=lambda x: x.visits)  # todo set real formula
        # best_node = max(
        #     node.subs, key=lambda x: (x.visits ** (1 / temperature)) / denominator
        # )
        return best_node.move

    @staticmethod
    def get_policy(node: Aux_MCTS.Node):
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # todo set to -1 illegal moves
        for n in node.subs:
            grid[n.move.x][n.move.y] = n.get_mean_value()
        return grid

    @staticmethod
    def choose_child(node: Aux_MCTS.Node, exploration) -> Aux_MCTS.Node:
        return max(node.subs, key=lambda x: x.interest(exploration))

    @staticmethod
    def expand(node: Aux_MCTS.Node, prior_moves, game: Game):
        for move in game.get_available_moves():
            new_game = game.__copy__()
            new_game.set_mark(move)
            node.subs.add(
                Aux_MCTS.Node(
                    state=new_game.get_state(),
                    prior=get_move_prior(prior_moves, move),
                    parent=node,
                    move=move,
                )
            )
        node.expanded = True

    @staticmethod
    def backprop(node: Aux_MCTS.Node, res):
        while node:
            node.value += res
            node.visits += 1
            node = node.parent
