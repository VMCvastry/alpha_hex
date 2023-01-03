from __future__ import annotations

import logging
import time
from collections import Counter

from game import Game
from utils.flip_board import flip_correct_state, flip_correct_point
from .mcst_aux import Aux_MCTS
from variables import *


# todo question page 26 alpha go zero


class MCTS:
    def __init__(
        self,
        network,
        init_state,
        player,
        exploration=None,
        temperature=None,
        simulations_cap=None,
        time_cap=None,
    ):
        # todo temperature ->0
        self.network = network
        self.player = player
        self.graph: Aux_MCTS.Node = Aux_MCTS.Node(
            state=flip_correct_state(init_state, player),
            prior=-1,
            parent=None,
            move=None,
        )
        self.stage, total_stages = Game(init_state, player).get_stage()
        print(f"Stage: {self.stage}/{total_stages}")
        # todo dirichlet noise on first move
        self.total_simulations = 0
        self.start_time = time.time()
        if exploration is None:
            exploration = EXPLORATION_PARAMETER
        if temperature is None:
            temperature = TEMPERATURE
        if simulations_cap is None:
            simulations_cap = SIMULATIONS_CAP
        self.simulation_cap = simulations_cap
        self.temperature = temperature
        self.exploration = exploration

    def search(self) -> tuple[Game.Move, list[list]]:
        while (
            self.total_simulations < self.simulation_cap
            and time.time() - self.start_time < TIME_CAP
        ):
            self.step()
            # logging.debug(Aux_MCTS.pick_best_move(self.graph, self.temperature))
        Aux_MCTS.normalize_layer(self.graph)
        move = Aux_MCTS.pick_best_move(
            self.graph, self.temperature if self.stage > 1 else 4 - self.stage
        )  # To recognize first and second move
        move.mark = self.player
        return (
            Game.Move(*flip_correct_point(move.x, move.y, self.player), self.player),
            flip_correct_state(
                Aux_MCTS.get_policy(self.graph), self.player, flip_sign=False
            ),
        )

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
        turn_player = node.get_next_player()
        game = Game(node.state, player=turn_player)
        if game.winner is not None:
            return game.winner
        moves, outcome = self.network.poll(node.state, turn_player)
        Aux_MCTS.expand(node, moves, game)
        return turn_player * outcome
