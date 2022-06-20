from __future__ import annotations

import copy
import random

from utils.logger import logging
from variables import GRID_SIZE


class Game:
    def __init__(self, board: list = None, player=1):
        self.board: list[list[Game.BoardCell]]
        self.winner = None
        self.player = player
        if board is None:
            self.board = [
                [Game.BoardCell(j, i) for i in range(GRID_SIZE)]
                for j in range(GRID_SIZE)
            ]
        else:
            self.board = [
                [Game.BoardCell(j, i, cell) for i, cell in enumerate(row)]
                for j, row in enumerate(board)
            ]
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.board[i][j].neighbours = self.get_neighbours(i, j)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                component_value = self.board[i][j].edge_connection()
                if component_value and self.propagate_component(i, j, component_value):
                    logging.critical("cloned finished game")
                    self.winner = self.player * -1
        if self.check_if_stale():
            self.winner = self.player * -1

    def __copy__(self):
        logging.critical(" Should not clone")
        raise NotImplementedError

    def get_state(self):
        return [[cell.value if cell else 0 for cell in row] for row in self.board]

    class Move:
        def __init__(self, x, y, player):
            self.x: int = x
            self.y: int = y
            self.mark = player

        def __str__(self):
            return "({}, {}) by {}".format(self.x, self.y, self.mark)

    class BoardCell:
        def __init__(self, x, y, value=0):
            self.x = x
            self.y = y
            self.value = value
            self.neighbours: list[Game.BoardCell] = []
            self.component = None

        def edge_connection(self):
            if self.value == 0:
                return False
            if self.x == 0 and self.value == 1:
                return "NORD"
            if self.x == GRID_SIZE - 1 and self.value == 1:
                return "SUD"

            if self.y == 0 and self.value == -1:
                return "EST"
            if self.y == GRID_SIZE - 1 and self.value == -1:
                return "OVEST"
            for n in self.neighbours:
                if n.component and self.value == n.value:
                    return n.component
            return False

        def __str__(self):
            return str(self.value)

    def propagate_component(self, x, y, component):
        cell = self.board[x][y]
        assert cell.value != 0
        cell.component = component
        for n in cell.neighbours:
            if n.component is None and n.value == cell.value:
                return self.propagate_component(n.x, n.y, component)
            if (
                n.component is not None
                and n.value == cell.value
                and n.component != component
            ):
                # print(n.component, component)
                return True

    def set_mark(self, move: Move):
        assert move.mark == self.player
        self.board[move.x][move.y].value = move.mark
        component_value = self.board[move.x][move.y].edge_connection()
        if component_value and self.propagate_component(
            move.x, move.y, component_value
        ):
            return move.mark
        if self.check_if_stale():
            # logging.debug("{} wins!".format(winner))
            return self.player
        self.player = -1 * self.player

    def get_marked_state(self, move: Move) -> list[list[int]]:
        state = self.get_state()
        state[move.x][move.y] = move.mark
        return state

    def check_if_stale(self) -> bool:

        if all(
            (
                self.board[x][y].value != 0
                for x in range(GRID_SIZE)
                for y in range(GRID_SIZE)
            )
        ):
            return True  # the player placing the last stone wins
        return False

    def get_available_moves(self) -> set[Game.Move]:
        moves = set()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.board[i][j] and self.board[i][j].value == 0:
                    moves.add(Game.Move(i, j, self.player))
        return moves

    def get_neighbours(self, x, y) -> list[Game.BoardCell]:
        neighbours = []
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if (
                    0 <= i < GRID_SIZE
                    and 0 <= j < GRID_SIZE
                    and (i, j) != (x, y)
                    and (i, j) != (x + 1, y + 1)
                    and (i, j) != (x - 1, y - 1)
                    and (i, j) != (x + 1, y - 1)
                    and (i, j) != (x - 1, y + 1)
                    and self.board[i][j]
                ):
                    if self.board[i][j]:
                        neighbours.append(self.board[i][j])
        return neighbours

    def __repr__(self):
        return (
            "\n"
            + "\n".join(
                [
                    str(
                        [
                            self.board[x][y].value if self.board[x][y] else 0
                            for y in range(GRID_SIZE)
                        ]
                    )
                    for x in range(GRID_SIZE)
                ]
            )
            + "\n"
        )


if __name__ == "__main__":

    def turn(game, move):
        winner = game.set_mark(move)
        logging.info(game)
        if winner is not None:
            logging.info("{} wins!".format(winner))
            exit(1)

    game = Game(player=1)
    while 1:
        logging.info("Player 1:")
        # logging.info([str(g) for g in game.get_available_moves()])
        # x, y = map(int, input(":").split(" "))
        # move = Game.Move(x, y, 1)
        move = random.choice(list(game.get_available_moves()))
        turn(game, move)

        logging.info("Player -1:")
        # x, y = map(int, input(":").split(" "))
        # move = Game.Move(x, y, -1)
        move = random.choice(list(game.get_available_moves()))
        turn(game, move)
