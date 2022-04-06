from __future__ import annotations

import logging
import random
import numpy as np
from game import Game


class Aux_MCTS:
    exploration_parameter = np.sqrt(2)

    class Node:
        def __init__(self, state, parent: Aux_MCTS.Node | None, move: Game.Move = None):
            self.state = state
            self.value = 0
            self.visits = 0.0000001
            self.explored = False
            self.expanded = False
            self.subs = set()
            self.parent: Aux_MCTS.Node = parent
            self.move = move

        def get_next_player(self):
            if not self.move:
                return 1
            else:
                return -1 * self.move.mark

        def ucb1(self, total_simulations):
            return self.value / self.visits + Aux_MCTS.exploration_parameter * np.sqrt(
                np.log(total_simulations) / self.visits
            )

        def __repr__(self):
            return f"{self.move},value {self.value}, visits={round(self.visits)}"

    @staticmethod
    def pick_move(node: Aux_MCTS.Node):
        logging.info([str(n) for n in node.subs])
        best_node = max(node.subs, key=lambda x: x.value / x.visits)
        return best_node.move

    @staticmethod
    def expand(node: Aux_MCTS.Node):
        game = Game(node.state)
        for move in game.get_available_moves():
            new_game = game.__copy__()
            new_game.set_mark(move)
            node.subs.add(Aux_MCTS.Node(new_game.get_state(), node, move))
        if not node.subs:
            print("a")
        node.expanded = True

    @staticmethod
    def mark_exploration(node: Aux_MCTS.Node):
        if node.expanded and all([n.explored for n in node.subs]):
            node.explored = True

    @staticmethod
    def simulate(node: Aux_MCTS.Node) -> int:
        game = Game(node.state, player=node.get_next_player())
        winner = game.check_tic_tac_toe()
        if winner is not None:
            node.expanded = True
            return winner
        while True:
            move = random.choice(list(game.get_available_moves()))
            winner = game.set_mark(move)
            if winner is not None:
                return winner

    @staticmethod
    def backprop(node: Aux_MCTS.Node, res):
        while node:
            Aux_MCTS.mark_exploration(node)
            node.value += res
            node.visits += 1
            node = node.parent
