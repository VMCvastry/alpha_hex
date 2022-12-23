from variables import GRID_SIZE, HEX_GRID_SIZE


# Flips a hex board and if 'flip_sign' is True, flips the sign of the values
def flip(board: list[list[int]], flip_sign=True) -> list[list[int]]:
    # [print(x) for x in board]
    new_board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    values = []
    for i, row in enumerate(board):
        if not (i % 3):
            v = []

            for j in range(i // 3, GRID_SIZE - (HEX_GRID_SIZE - 1) + i // 3, 2):
                v.append(row[j])
            values.append(v)
    # [print(x) for x in values]
    new_values = [[0 for _ in range(HEX_GRID_SIZE)] for _ in range(HEX_GRID_SIZE)]
    for i in range(len(values)):
        for j in range(len(values[i])):
            new_values[i][j] = values[j][i]
    # [print(x) for x in new_values]
    for i, row in enumerate(new_board):
        if not (i % 3):
            jj = 0
            for j in range(i // 3, GRID_SIZE - (HEX_GRID_SIZE - 1) + i // 3, 2):
                row[j] = new_values[i // 3][jj] * (
                    -1 if flip_sign else 1
                )  # -1 to flip the sign
                jj += 1
    # [print(x) for x in new_board]
    return new_board


# Flips a board if the player is -1
def flip_correct_state(
    state: list[list[int]], player, flip_sign=True
) -> list[list[int]]:
    return flip(state, flip_sign) if player == -1 else state


def flip_point(x, y):
    new_y = (x // 3) * 2
    new_x = ((y - x // 3) // 2) * 3
    return new_x, new_x // 3 + new_y


# Flips a move if the player is -1
def flip_correct_point(x, y, player):
    # print(x, y)
    # print(flip_point(x, y) if player == -1 else (x, y))
    return flip_point(x, y) if player == -1 else (x, y)


if __name__ == "__main__":
    HEX_GRID_SIZE = 2
    GRID_SIZE = 3 * HEX_GRID_SIZE - 2

    board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    board[0][0] = 1
    board[0][2] = 2
    board[3][1] = 3
    board[3][3] = 4
    board[2][2] = 5
    board = [
        ["0", " ", "0", " ", "0", " ", "0", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", "0", " ", "0", " ", "0", " ", "0", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", "1", " ", "0", " ", "0", " ", "0", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", "0", " ", "0", " ", "-1", " ", "0"],
    ]
    [print(x) for x in board]
    move = (3, 1)
    print(move)
    f = flip(board)
    f_m = flip_correct_point(*move)
    [print(x) for x in f]
    print(f_m)
    ff_m = flip_correct_point(*f_m)
    [print(x) for x in flip(f)]
    print(ff_m)
