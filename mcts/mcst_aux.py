from __future__ import annotations

from logger import logging
import random
import numpy as np
from game import Game
from variables import *


def get_move_prior(priors, move: Game.Move):
    # print(priors)
    # print(priors[move.x][move.y])
    return float(priors[move.x][move.y])


# todo mcts policy not a prob distirbution
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
            self.normalized_value = None

        def get_mean_value(self):
            return self.value / (self.visits + 0.00001)

        def get_normalized_worth(self):
            assert self.normalized_value is not None
            return self.normalized_value

        def get_next_player(self):
            if not self.move:
                return 1
            else:
                return -1 * self.move.mark

        def layer_visits(self):
            subs = self.parent.subs
            return sum(s.visits for s in subs)

        def interest(self, exploration):
            return self.get_mean_value() + exploration * (
                self.prior * np.sqrt(self.layer_visits()) / (self.visits + 1)
            )  # todo why prior

        def __repr__(self):
            return f"{self.move}, mean_value={self.get_mean_value()},value {self.value}, visits={round(self.visits)}, interest={self.interest(100)}"

    @staticmethod
    def flip_state(state, player):
        return [[player * s for s in row] for row in state]

    @staticmethod
    def pick_best_move(node: Aux_MCTS.Node, temperature):
        logging.debug("possible MCTS moves \n" + "\n".join([str(n) for n in node.subs]))
        # logging.debug(
        #     f"possible MCTS moves {[n.get_normalized_value() for n in node.subs]}"
        # )
        # denominator = node.layer_visits() ** (1 / temperature)
        # best_node = max(node.subs, key=lambda x: x.visits)  # todo set real formula
        # best_node = max(
        #     node.subs, key=lambda x: (x.visits ** (1 / temperature)) / denominator
        # )
        if temperature:
            best_node = random.choices(
                list(node.subs), weights=[n.get_normalized_worth() for n in node.subs]
            )[0]
        else:
            best_node = max(node.subs, key=lambda x: x.get_normalized_worth())
        return best_node.move

    @staticmethod
    def normalize_layer(node: Aux_MCTS.Node):
        # minimum = min(n.get_mean_value() for n in node.subs)
        # total = sum(n.get_mean_value() + abs(minimum) for n in node.subs)
        # for n in node.subs:
        #     if not total:  # only losing moves
        #         logging.debug("here checlk")
        #         n.normalized_value = 1 / (len(node.subs))
        #     else:
        #         n.normalized_value = (n.get_mean_value() + abs(minimum)) / total
        # logging.debug(f"moves {[str(n.move) for n in node.subs]}")
        # logging.debug(f"mean {[n.get_normalized_value() for n in node.subs]}")
        # minimum = min(n.value for n in node.subs)
        # total = sum(n.value + abs(minimum) for n in node.subs)
        # for n in node.subs:
        #     if not total:  # only losing moves
        #         logging.debug("here checlk")
        #         n.normalized_value = 1 / (len(node.subs))
        #     else:
        #         n.normalized_value = (n.value + abs(minimum)) / total
        # logging.debug(f"total {[n.get_normalized_value() for n in node.subs]}")
        total = sum(n.visits for n in node.subs)
        for n in node.subs:
            n.normalized_value = n.visits / total
        # logging.debug(f"visits {[n.get_normalized_value() for n in node.subs]}")

    @staticmethod
    def get_policy(node: Aux_MCTS.Node):
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # todo set to -1 illegal moves
        for n in node.subs:
            grid[n.move.x][n.move.y] = n.get_normalized_worth()
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
