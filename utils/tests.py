from __future__ import annotations
from utils.logger import logging
from mcts.mcst import MCTS
from game import Game, discard_rows_for_print
from net.trainer import Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def turn(game, move):
    game.set_mark(move)
    logging.info(game)
    if game.winner is not None:
        logging.info("{} wins!".format(game.winner))


#
# model = NET(2, HIDDEN_FEATURES, RESNET_DEPTH, VALUE_HEAD_SIZE).to(device)
# loss_fn = nn.MSELoss(reduction="mean")
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def test(trainer, game, player_turn):
    print(trainer.poll(game.get_state(), player_turn))

    player = MCTS(trainer, game.get_state(), player_turn, exploration=1.4)
    a, b = player.search()
    print(a)
    print(b)


player_turn = 1
# print(game)
# print("player turn: {}".format(player_turn))


# test(Trainer(model_name="NEW_NET_2022-04-21_09-32-57_BEST"), game, player_turn)
# test(Trainer(model_name="FIXED_NET_2022-05-13_11-12-33"), game, player_turn)
# test(Trainer(model_name="OLDEXP_FIXED_NET_2022-05-13_15-37-30"), game, player_turn)
# test(Trainer(model_name="NEWEXP_FIXED_NET_2022-05-13_15-37-30"), game, player_turn)
# #
# test(Trainer(model_name="NEWEXP_FIXED_NET_2022-05-13_16-47-13"), game, player_turn)
# test(Trainer(model_name="FIXED_NET_2022-05-14_19-50-16"), game, player_turn)
# cur = "FIXED_NET_2022-05-15_13-33-47"
# cur = "FIXED_NET_2022-05-31_23-09-06"
# print(Trainer(model_name=cur).poll(game.get_state(), player_turn))
# print(
#     Trainer(model_name=cur).poll(
#         Game([[0, -1, 1], [-1, -1, 1], [0, 1, 0]]).get_state(), player_turn
#     )
# )

#

# print(opt.poll([[1, None, None], [1, -1, None], [-1, 1, -1]]))

# exit()
# # new_trainer = Trainer(model_name="NEW_NET_2022-04-21_09-32-57_BEST")
new_trainer = Trainer(model_name="R_1_HEX_NET_2022-12-23_17-10-39")
new_trainer = Trainer(model_name="R_2_HEX_NET_2022-12-24_14-40-14")
new_trainer = Trainer(model_name="R_3_HEX_NET_2022-12-24_23-02-27")
new_trainer = Trainer(model_name="R_3_HEX_NET_2022-12-25_15-26-10")
new_trainer = Trainer(model_name="R_3_HEX_NET_2022-12-26_05-17-54")
new_trainer = Trainer(model_name="R_3_HEX_NET_2022-12-27_06-37-56")
game = Game(player=1)
if game.player == -1:
    x, y = map(int, input(":").split(" "))
    move = Game.Move(x * 3, y * 2 + x, -1)
    # move = random.choice(list(game.get_available_moves()))
    turn(game, move)
while 1:
    p, v = new_trainer.poll(game.get_state(), player_turn)
    [print(list(map("{:.2f}".format, x))) for x in discard_rows_for_print(p.tolist())]
    # logging.info(discard_rows_for_print(p.tolist()))
    logging.info(v)
    player = MCTS(
        new_trainer,
        game.get_state(),
        1,
        exploration=1.4,
        temperature=0,
        simulations_cap=1,
    )
    a, b = player.search()
    logging.info(a)
    logging.info(discard_rows_for_print(b))
    turn(game, a)

    x, y = map(int, input(":").split(" "))
    move = Game.Move(x * 3, y * 2 + x, -1)
    # move = random.choice(list(game.get_available_moves()))
    turn(game, move)

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
game = Game(g)
logging.info(new_trainer.poll(game.get_state(), 1))
player = MCTS(new_trainer, game.get_state(), 1, exploration=1.4, temperature=0)
a, b = player.search()
logging.info(a)
logging.info(b)
# game = Game([[1, 0, 0], [-1, -1, 0], [1, 0, -1]])
# game = Game([[0, 0, 0], [1, 0, -1], [-1, 0, 0]])
# logging.info(new_trainer.poll(game.get_state(), player_turn))
# logging.info(new_trainer2.poll(game.get_state(), player_turn))
# logging.info(new_trainer3.poll(game.get_state(), player_turn))
# logging.info(new_trainer3.poll(game.get_state(), player_turn))
# player = MCTS(new_trainer, game.get_state(), 1, exploration=1.4)
# a, b = player.search()
# logging.info(a)
# logging.info(b)
