from __future__ import annotations

import copy

N = 6


class Game:
    def __init__(self, board: list = None, player=1):
        if board is None:
            self.board = [[0] * N for i in range(N)]
        else:
            self.board = copy.deepcopy(board)
        self.player = player

    def __copy__(self):
        game = Game()
        game.board = copy.deepcopy(self.board)
        game.player = self.player
        return game

    def get_state(self):
        return copy.deepcopy(self.board)

    class Move:
        def __init__(self, x, y, player):
            self.x: int = x
            self.y: int = y
            self.mark = player

        def __str__(self):
            return "({}, {}) by {}".format(self.x, self.y, self.mark)

    def set_mark(self, move: Move):
        assert move.mark == self.player
        self.board[move.x][move.y] = move.mark
        winner = self.check_if_winner()
        if winner is not None:
            # logging.debug("{} wins!".format(winner))
            return winner
        self.player = -1 * self.player

    def check_if_winner(self) -> int | None:

        if all((self.board[x][y] != 0 for x in range(N) for y in range(N))):
            return self.player  # the player placing the last stone wins
        return None

    def get_available_moves(self) -> set[Game.Move]:
        moves = set()
        for i in range(N):
            for j in range(N):
                if self.board[i][j] == 0:
                    moves.add(Game.Move(i, j, self.player))
        return moves

    def __repr__(self):
        return (
            "\n"
            + "\n".join(
                [
                    str([self.board[x][y] if self.board[x][y] else 0 for y in range(N)])
                    for x in range(N)
                ]
            )
            + "\n"
        )
