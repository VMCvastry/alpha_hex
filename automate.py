from __future__ import annotations

import datetime
import time

from train_net import train_net
from self_play import run_self_play
from duel import find_best
import sys
from logger import logging
from drive_explorer import DriveExplorer
import os

# fix value
# speed duel


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
# model_name = "NEW_NET_2022-04-21_09-32-57"
# gen = 53
model_name = "FIXED_NET_2022-05-13_16-47-13"
gen = 61
model_name = ""
gen = 1
save_drive = False
COLAB = False
if len(sys.argv) > 1:
    COLAB = True
    save_drive = True
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
        gen = int(sys.argv[3])
if save_drive:
    drive = DriveExplorer(on_colab=COLAB)
    drive.retrieve_model(model_name)
    drive.retrieve_training_data("FIXED_61")
    drive.retrieve_training_data("FIXED_60")
    drive.retrieve_training_data("FIXED_62")
    drive.retrieve_training_data("FIXED_63")
    drive.retrieve_training_data("FIXED_64")
    time.sleep(3)
print(f"model_name: {model_name}", gen)
total_cycles = gen
datasets = [
    "gen23",
    "FIXED_53",
    "FIXED_60",
    "FIXED_61",
    "FIXED_62",
    "FIXED_63",
    "FIXED_64",
]
temp = 1
while 1:
    logging.info(f"GEN: {gen}, model: {model_name}, total:{total_cycles}")
    if not temp:
        run_self_play(f"REBORN_{gen}", model_name)
    datasets = datasets[-5:]
    new_model_name = train_net(datasets, model_name)

    res = find_best(new_model_name, model_name)
    if res[1] > res[-1]:
        model_name = new_model_name
        if save_drive:
            drive.save_model(model_name)
            drive.save_training_data(datasets[-1])
        gen += 1
        datasets.append(f"REBORN_{gen}")
    else:
        logging.info("No improvement")
        if os.path.exists(f"models/{new_model_name}.pt"):
            os.remove(f"models/{new_model_name}.pt")
            logging.debug(f"deleted models/{new_model_name}.pt")
    total_cycles += 1
    temp = 0
