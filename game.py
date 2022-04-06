from __future__ import annotations

import copy
import logging


class Game:
    def __init__(self, board: list = None, player=1):
        if board is None:
            self.board = [[None] * 3 for i in range(3)]
        else:
            self.board = copy.deepcopy(board)
        self.player = player

    def __copy__(self):
        game = Game()
        game.board = copy.deepcopy(self.board)
        game.player = self.player
        return game

    def get_state(self):
        return self.board

    class Move:
        def __init__(self, x, y, player):
            self.x = x
            self.y = y
            self.mark = player

        def __str__(self):
            return "({}, {}) by {}".format(self.x, self.y, self.mark)

    def set_mark(self, move: Move):
        self.board[move.x][move.y] = move.mark
        winner = self.check_tic_tac_toe()
        if winner is not None:
            logging.debug("{} wins!".format(winner))
            return winner
        self.player = -1 * self.player

    def check_tic_tac_toe(self) -> int | None:
        # Check rows
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] is not None:
                return self.board[i][0]
        # Check columns
        for i in range(3):
            if self.board[0][i] == self.board[1][i] == self.board[2][i] is not None:
                return self.board[0][i]
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] is not None:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] is not None:
            return self.board[0][2]
        if all((self.board[x][y] is not None for x in range(3) for y in range(3))):
            return 0
        return None

    def get_available_moves(self) -> set[Game.Move]:
        moves = set()
        for i in range(3):
            for j in range(3):
                if self.board[i][j] is None:
                    moves.add(Game.Move(i, j, self.player))
        return moves

    def __repr__(self):
        return (
            "\n"
            + "\n".join(
                [
                    str([self.board[x][y] if self.board[x][y] else 0 for y in range(3)])
                    for x in range(3)
                ]
            )
            + "\n"
        )
