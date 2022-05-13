from __future__ import annotations

import datetime

from train_net import train_net
from self_play import run_self_play
from duel import find_best
import sys
from logger import logging
import os

# fix value
# speed duel


def save_colab(on_colab: bool):
    if on_colab:
        os.popen(
            f"cp -r './models/{model_name}.pt' '/content/gdrive/My Drive/TRIS/models'"
        )
        os.popen(
            f"cp -r './training_data/{datasets[-1]}_policies.pkl' '/content/gdrive/My Drive/TRIS/training_data'"
        )
        os.popen(
            f"cp -r './training_data/{datasets[-1]}_states.pkl' '/content/gdrive/My Drive/TRIS/training_data'"
        )
        os.popen(
            f"cp -r './training_data/{datasets[-1]}_values.pkl' '/content/gdrive/My Drive/TRIS/training_data'"
        )
        os.popen(
            f"cp -r './logs.log' '/content/gdrive/My Drive/TRIS/logs{datetime.datetime.now()}.log'"
        )


# model_name = "NET_2022-04-14_09-29-42"
# gen = 8
# model_name = "NET_2022-04-16_11-16-31"
# gen = 23
# model_name = "NET_2022-04-16_18-02-38"
# gen = 28
# model_name = "NET_2022-04-17_09-05-35"
# gen = 26
# model_name = "NET_2022-04-18_08-02-57"
# gen = 35
model_name = "NEW_NET_2022-04-21_09-32-57"
gen = 53
COLAB = False
if len(sys.argv) > 1:
    COLAB = True
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
        gen = int(sys.argv[3])
print(f"model_name: {model_name}", gen)
total_cycles = gen
datasets = ["gen8", "gen23"]
datasets = [f"FIXED_{gen}"]
temp = 1
while 1:
    logging.info(f"GEN: {gen}, model: {model_name}, total:{total_cycles}")
    if not temp:
        run_self_play(f"FIXED_{gen}", model_name)
    datasets = datasets[-5:]
    new_model_name = train_net(datasets, model_name)

    res = find_best(new_model_name, model_name)
    if res[1] > res[-1]:
        model_name = new_model_name
        gen += 1
        datasets.append(f"FIXED_{gen}")
        save_colab(COLAB)
    else:
        logging.info("No improvement")
        if os.path.exists(f"models/{new_model_name}.pt"):
            os.remove(f"models/{new_model_name}.pt")
            logging.debug(f"deleted models/{new_model_name}.pt")
    total_cycles += 1
    temp = 0
