from __future__ import annotations

HEX_GRID_SIZE = 7
GRID_SIZE = 3 * HEX_GRID_SIZE - 2
OUTPUT_LABEL = "R_5_HEX"

N_EPOCHS = 300
BATCH_SIZE = 128
HIDDEN_FEATURES = 128
RESNET_DEPTH = 30
VALUE_HEAD_SIZE = 64
TEST_RATIO = 0.2

EXPLORATION_PARAMETER = 2
TEMPERATURE = 1
SIMULATIONS_CAP = 100
TIME_CAP = 100000000000

N_GAMES = 100

N_GAMES_DUEL = 30

LEARNING_RATE = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4  # todo is equal to l2 regularization?
# dropout = 0.2

PLOT = 1
DEBUG = 0