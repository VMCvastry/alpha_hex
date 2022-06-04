from __future__ import annotations
from utils.logger import logging
from mcts.mcst import MCTS
from game import Game
from net.trainer import Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def turn(game, move):
    game.set_mark(move)
    logging.info(game)
    if game.check_if_winner() is not None:
        logging.info("{} wins!".format(game.check_if_winner()))


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


game = Game([[1, 0, 0], [1, -1, 0], [-1, 1, -1]])
game = Game([[0, 1, 1], [-1, 0, 0], [0, 0, 0]])
game = Game([[1, 0, 0], [-1, 1, 0], [-1, -1, 0]])
game = Game([[0, 0, 1], [0, -1, -1], [0, 0, 1]])
# game = Game([[-1, 1, 1], [-1, 0, 0], [1, 0, 0]])
# game = Game([[-1, 0, -1], [1, 0, 0], [-1, 1, 1]])
# game = Game([[-1, 0, 0], [0, 0, 1], [0, 0, 0]])


game = Game([[0, -1, 1], [-1, -1, 1], [0, 1, 0]])
game = Game([[-1, 0, -1], [1, 0, 0], [1, 1, -1]])

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
new_trainer = Trainer(model_name="REBORN_NET_2022-05-31_23-52-28")  # WRONG TRIS
new_trainer2 = Trainer(model_name="REBORN_NET_2022-06-01_12-11-28")
new_trainer3 = Trainer(model_name="REBORN_NET_2022-06-01_12-35-11")
new_trainer4 = Trainer(model_name="REBORN_NET_2022-06-02_17-35-19")
game = Game(player=1)
while 1:
    logging.info(new_trainer.poll(game.get_state(), player_turn))
    logging.info(new_trainer2.poll(game.get_state(), player_turn))
    logging.info(new_trainer3.poll(game.get_state(), player_turn))
    logging.info(new_trainer4.poll(game.get_state(), player_turn))
    player = MCTS(new_trainer4, game.get_state(), 1, exploration=1.4)
    a, b = player.search()
    logging.info(a)
    logging.info(b)
    turn(game, a)

    x, y = map(int, input(":").split(" "))
    move = Game.Move(x, y, -1)
    # move = random.choice(list(game.get_available_moves()))
    turn(game, move)


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
