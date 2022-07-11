from __future__ import annotations

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
        if self.check_if_won(self.player * -1):
            self.winner = self.player * -1

        elif self.check_if_stale():
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
            self.visited = False

        def __str__(self):
            return str(self.value)

    def reset_cells(self):
        for row in self.board:
            for cell in row:
                cell.visited = False

    def is_connected(self, player, x, y):
        queue = [self.board[x][y]]
        while queue:
            cell = queue[0]
            queue = queue[1:]
            cell.visited = True
            neighbours = self.get_neighbours(cell.x, cell.y)
            for n in neighbours:
                if not n.visited and n.value == player:
                    queue.append(n)
                    if (n.x == (GRID_SIZE - 1) and player == 1) or (
                        n.y == (GRID_SIZE - 1) and player == -1
                    ):
                        return True

    def check_if_won(self, player):
        self.reset_cells()
        if player == 1:
            for y in range(GRID_SIZE):
                if self.board[0][y].value == 1 and self.is_connected(player, 0, y):
                    return True
        else:
            for x in range(GRID_SIZE):
                if self.board[x][0].value == -1 and self.is_connected(player, x, 0):
                    return True

    def set_mark(self, move: Move) -> None:
        assert move.mark == self.player
        self.board[move.x][move.y].value = move.mark
        if self.check_if_won(self.player):
            self.winner = self.player
            return
        if self.check_if_stale():
            # logging.debug("{} wins!".format(winner))
            self.winner = self.player
            return
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
        game.set_mark(move)
        logging.info(game)
        if game.winner is not None:
            logging.info("{} wins!".format(game.winner))
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
