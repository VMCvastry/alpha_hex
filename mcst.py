from __future__ import annotations

import time

from mcst_aux import Aux_MCTS

SIMULATIONS_CAP = 100
TIME_CAP = 100000000000


class MCTS:
    def __init__(self, init_state):
        self.graph: Aux_MCTS.Node = Aux_MCTS.Node(init_state, parent=None)
        Aux_MCTS.expand(self.graph)
        self.total_simulations = 0
        self.start_time = time.time()

    def search(self):
        while (
            not self.graph.explored
            and self.total_simulations < SIMULATIONS_CAP
            and time.time() - self.start_time < TIME_CAP
        ):
            self.step()
        return Aux_MCTS.pick_move(self.graph)

    def choose_child(self, node: Aux_MCTS.Node) -> Aux_MCTS.Node:
        return max(
            (n for n in node.subs if not n.explored),
            key=lambda x: node.get_next_player() * x.ucb1(self.total_simulations),
        )

    def step(self):
        self.total_simulations += 1
        next_node = self.graph
        while True:
            next_node = self.choose_child(next_node)
            if not next_node.subs:
                if next_node.visits < 1:
                    break
                else:
                    Aux_MCTS.expand(next_node)
        outcome = Aux_MCTS.simulate(next_node)
        Aux_MCTS.backprop(next_node, outcome)
