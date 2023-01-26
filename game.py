from __future__ import annotations

import random

from utils.logger import logging
from variables import GRID_SIZE, HEX_GRID_SIZE

"""
Magic number:
top squares num 8, so 8%(6+1)=1 ->start
bottom squares num 9, so 9%(6+1)=2 ->end
left squares num 6, so 6%(6-1)=1 ->start
right squares num 7, so 7%(6-1)=2 ->end
top left num 36
top right num 22
bot left num 16
bot right num 37
"""


def discard_rows_for_print(state):
    return [s for i, s in enumerate(state) if i % 3 == 0]


class Game:
    def __init__(self, board: list = None, player=1):
        self.board: list[list[Game.BoardCell]]
        self.winner = None
        self.player = player
        self.build_board()
        self.first_move = True
        if board is not None:
            count = 0
            for i, row in enumerate(board):
                for j, cell in enumerate(row):
                    self.board[i][j].value = cell
                    count += abs(cell)
            if count != 0:
                self.first_move = False

        if self.check_if_won(self.player * -1):
            self.winner = self.player * -1

        elif self.check_if_stale():
            self.winner = self.player * -1

    def __copy__(self):
        logging.critical(" Should not clone")
        raise NotImplementedError

    def get_state(self) -> list[list[int]]:
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
            self.valid = False
            self.magic = 0

        def __str__(self):
            return str(self.value)

    def build_board(self):
        self.board = [
            [Game.BoardCell(j, i) for i in range(GRID_SIZE)] for j in range(GRID_SIZE)
        ]
        for i, row in enumerate(self.board):
            if not (i % 3):
                for j in range(i // 3, GRID_SIZE - (HEX_GRID_SIZE - 1) + i // 3, 2):
                    row[j].valid = True
                    if i == 0 and j == 0:
                        row[j].magic = 36
                    elif i == 0 and j == GRID_SIZE - (HEX_GRID_SIZE - 1) - 1:
                        row[j].magic = 22
                    elif i == GRID_SIZE - 1 and j == (HEX_GRID_SIZE - 1):
                        row[j].magic = 16
                    elif i == GRID_SIZE - 1 and j == GRID_SIZE - 1:
                        row[j].magic = 37
                    elif i == 0:
                        row[j].magic = 8
                    elif i == GRID_SIZE - 1:
                        row[j].magic = 9
                    elif j == i // 3:
                        row[j].magic = 6
                    elif j == GRID_SIZE - (HEX_GRID_SIZE - 1) + i // 3 - 1:
                        row[j].magic = 7

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
                    if n.magic % (6 + player) == 2:
                        return True

    def check_if_won(self, player):
        self.reset_cells()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (
                    self.board[i][j].value == player
                    and self.board[i][j].magic % (6 + player) == 1
                ):
                    if self.is_connected(player, i, j):
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
        if self.first_move:
            self.board[GRID_SIZE - 1][0].value = 2  # to mark second move
        else:
            self.board[GRID_SIZE - 1][0].value = 0
        self.first_move = False

    def get_marked_state(self, move: Move) -> list[list[int]]:
        state = self.get_state()
        state[move.x][move.y] = move.mark
        if self.first_move:
            state[GRID_SIZE - 1][0] = 2  # to mark second move
        else:
            state[GRID_SIZE - 1][0] = 0
        return state

    def check_if_stale(self) -> bool:

        if all(
            (
                self.board[x][y].value != 0 or not self.board[x][y].valid
                for x in range(GRID_SIZE)
                for y in range(GRID_SIZE)
            )
        ):
            return True  # the player placing the last stone wins
        return False

    def get_available_moves(self) -> set[Game.Move]:
        moves = set()
        second_move = self.board[GRID_SIZE - 1][0].value == 2
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (
                    # self.board[i][j] and
                    self.board[i][j].valid
                    and (self.board[i][j].value == 0 or second_move)
                ):
                    moves.add(Game.Move(i, j, self.player))
        return moves

    def get_neighbours(self, i, j) -> list[Game.BoardCell]:
        neighbours = []
        possible = [
            (i, j - 2),
            (i, j + 2),
            (i + 3, j - 1),
            (i + 3, j + 1),
            (i - 3, j - 1),
            (i - 3, j + 1),
        ]
        for p_i, p_j in possible:
            if (
                0 <= p_j < GRID_SIZE
                and 0 <= p_i < GRID_SIZE
                and self.board[p_i][p_j].valid
            ):
                neighbours.append(self.board[p_i][p_j])
        return neighbours

    def get_stage(self) -> tuple[int, int]:
        return (
            sum([sum([abs(cell.value) for cell in row]) for row in self.board]),
            HEX_GRID_SIZE**2,
        )

    def __repr__(self):
        # return (
        #     "\n"
        #     + "\n".join(
        #         discard_rows_for_print(
        #             [
        #                 str(
        #                     [
        #                         str(self.board[x][y].value)
        #                         if self.board[x][y].valid
        #                         else " "
        #                         for y in range(GRID_SIZE)
        #                     ]
        #                 )
        #                 for x in range(GRID_SIZE)
        #             ]
        #         )
        #     )
        #     + "\n"
        # )

        raw = discard_rows_for_print(
            [
                [
                    self.board[x][y].value if self.board[x][y].valid else " "
                    for y in range(GRID_SIZE)
                ]
                for x in range(GRID_SIZE)
            ]
        )
        string = ""
        for row in raw:
            string += "\n"
            for cell in row:
                if cell == 0:
                    string += "."
                elif cell == 1:
                    string += "X"
                elif cell == -1:
                    string += "O"
                else:
                    string += " "
        return string + f"{self.first_move},{self.board[GRID_SIZE - 1][0].value}"


if __name__ == "__main__":

    def turn(game, move):
        game.set_mark(move)
        logging.info(game)
        if game.winner is not None:
            logging.info("{} wins!".format(game.winner))
            exit(1)

    # game = Game(player=1)
    # while 1:
    #     logging.info("Player 1:")
    #     # logging.info([str(g) for g in game.get_available_moves()])
    #     # x, y = map(int, input(":").split(" "))
    #     # move = Game.Move(x, y, 1)
    #     move = random.choice(list(game.get_available_moves()))
    #     turn(game, move)
    #
    #     logging.info("Player -1:")
    #     # x, y = map(int, input(":").split(" "))
    #     # move = Game.Move(x, y, -1)
    #     move = random.choice(list(game.get_available_moves()))
    #     turn(game, move)
    g = [
        [0, 0, -1, 0, -1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 1, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 1, 0, 1, 0, -1],
    ]
    g = [
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, -1, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 1, 0, 0],
    ]
    g = [
        [0, 0, 0, 0, -1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 1, 0, 1, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, -1, 0, 1, 0, 0],
    ]
    p = Game(player=1, board=g)
    print(p)
    print(p.winner)

    def rotate_left(board: list[list[int]]):
        return [list(reversed(row)) for row in zip(*board)]

    gg = rotate_left(rotate_left(g))
    pp = Game(player=1, board=gg)
    print(pp)
    # p.set_mark(Game.Move(9, 5, -1))
    # print(p)
    # print(p.winner)
