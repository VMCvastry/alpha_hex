from __future__ import annotations

from utils.logger import logging
import random
import numpy as np
from game import Game
from variables import GRID_SIZE


def get_move_prior(priors, move: Game.Move):
    # print(priors)
    # print(priors[move.x][move.y])
    return float(priors[move.x][move.y])


class Aux_MCTS:
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
            self.interest = None
            self.posterior = None

        def get_mean_value(self):
            return self.value / self.visits if self.visits > 0 else 0

        # def get_mean_value(self):
        #     return -1 * self.get_next_player() * self.value / (self.visits + 0.00001)
        #
        # def get_normalized_worth(self):
        #     assert self.normalized_value is not None
        #     return self.normalized_value

        def get_next_player(self):
            if not self.move:
                return 1
            else:
                return -1 * self.move.mark

        # def layer_visits(self):
        #     subs = self.parent.subs
        #     return sum(s.visits for s in subs)

        # def interest(
        #     self, exploration
        # ):
        #     return self.get_mean_value() + exploration * (
        #         self.prior * np.sqrt(self.layer_visits() + 1) / (self.visits + 1)
        #     )

        def __repr__(self):
            return f"{self.move}, mean_value={self.get_mean_value()},value {self.value}, visits={round(self.visits)}"

    @staticmethod
    def gen_posterior(node: Aux_MCTS.Node, temperature):
        if not temperature:
            for sub in node.subs:
                sub.posterior = 0
            best_node = max(node.subs, key=lambda x: x.visits)
            best_node.posterior = 1
            return
        layer_visits = sum(s.visits for s in node.subs)
        denominator = layer_visits ** (1 / temperature)
        for sub in node.subs:
            sub.posterior = sub.visits ** (1 / temperature) / denominator

    @staticmethod
    def pick_best_move(node: Aux_MCTS.Node):
        logging.debug("possible MCTS moves \n" + "\n".join([str(n) for n in node.subs]))
        logging.debug(f"possible MCTS moves {[n.posterior for n in node.subs]}")
        # best_node = max(node.subs, key=lambda x: x.visits)  # todo set real formula
        best_node = max(node.subs, key=lambda x: x.posterior)
        # if temperature:
        # #     best_node = random.choices(
        # #         list(node.subs), weights=[n.get_normalized_worth() for n in node.subs]
        # #     )[0]
        # #     if temperature == 2 and random.random() < 0.2:
        # #         best_node = random.choice(list(node.subs))
        #     if temperature == 3 and random.random() < 0.5:
        #         best_node = random.choice(list(node.subs))
        #     elif temperature == 4 and random.random() < 0.8:
        #         best_node = random.choice(list(node.subs))
        # if not temperature:
        #     best_node = max(node.subs, key=lambda x: x.visits)
        return best_node.move

    @staticmethod
    def get_policy(node: Aux_MCTS.Node):
        grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for n in node.subs:
            grid[n.move.x][n.move.y] = n.posterior
        return grid

    @staticmethod
    def choose_child(node: Aux_MCTS.Node, exploration) -> Aux_MCTS.Node:
        layer_visits = sum(s.visits for s in node.subs)
        for n in node.subs:
            n.interest = n.get_mean_value() + exploration * (
                n.prior / (n.visits + 1)
            ) * np.sqrt(layer_visits)
        return max(node.subs, key=lambda x: x.interest)

    @staticmethod
    def expand(node: Aux_MCTS.Node, prior_moves, game: Game):
        for move in game.get_available_moves():
            new_state = game.get_marked_state(move)
            node.subs.add(
                Aux_MCTS.Node(
                    state=new_state,
                    prior=get_move_prior(prior_moves, move),
                    parent=node,
                    move=move,
                )
            )
        total = sum(n.prior for n in node.subs)
        for n in node.subs:
            n.prior = n.prior / total
        node.expanded = True

    @staticmethod
    def backprop(node: Aux_MCTS.Node, res):
        while node:
            node.value += res
            node.visits += 1
            node = node.parent
