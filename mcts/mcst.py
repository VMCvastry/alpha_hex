from __future__ import annotations

import time
from collections import Counter

from game import Game
from .mcst_aux import Aux_MCTS
from variables import *

SIMULATIONS_CAP = 100
TIME_CAP = 100000000000

# todo question page 26 alpha go zero


class MCTS:
    def __init__(self, network, init_state, player, exploration=None, temperature=None):
        # todo temperature ->0
        self.network = network
        self.player = player
        self.graph: Aux_MCTS.Node = Aux_MCTS.Node(
            state=Aux_MCTS.flip_state(init_state, player),
            prior=-1,
            parent=None,
            move=None,
        )
        # todo dirichlet noise on first move
        self.total_simulations = 0
        self.start_time = time.time()
        if exploration is None:
            exploration = EXPLORATION_PARAMETER
        if temperature is None:
            temperature = TEMPERATURE
        self.temperature = temperature
        self.exploration = exploration

    def search(self) -> tuple[Game.Move, list[list]]:
        while (
            self.total_simulations <= SIMULATIONS_CAP
            and time.time() - self.start_time < TIME_CAP
        ):
            self.step()
            # print(Aux_MCTS.pick_best_move(self.graph))
        Aux_MCTS.normalize_layer(self.graph)
        move = Aux_MCTS.pick_best_move(self.graph, self.temperature)
        move.mark = self.player
        return move, Aux_MCTS.get_policy(self.graph)

    def step(self):
        self.total_simulations += 1
        next_node = self.graph
        while True:
            if (
                next_node.visits < 1 or not next_node.subs
            ):  # todo check "or not next_node.subs", should be good
                break
            next_node = Aux_MCTS.choose_child(next_node, self.exploration)
        outcome = self.simulate(next_node)
        Aux_MCTS.backprop(next_node, outcome)

    def simulate(self, node: Aux_MCTS.Node) -> int:
        game = Game(node.state, player=node.get_next_player())
        winner = game.check_if_winner()
        if winner is not None:  # todo check, should be good
            return winner
        moves, outcome = self.network.poll(node.state, node.get_next_player())
        Aux_MCTS.expand(node, moves, game)
        return node.get_next_player() * outcome
