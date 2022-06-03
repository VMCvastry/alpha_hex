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
model_name = "REBORN_NET_2022-05-31_23-52-28"
gen = 13
total_cycles = 27
datasets = [
    "gen23",
    "FIXED_53",
    "FIXED_60",
    "FIXED_61",
    "FIXED_62",
    "FIXED_63",
    "FIXED_64",
]
datasets = [
    # "FIXED_61",
    # "FIXED_62",
    # "REBORN_8",
    "REBORN_9",
    "REBORN_10",
    "REBORN_11",
    "REBORN_12",
]
gen = 0
total_cycles = 0
datasets = ["WINNER_0"]
model_name = ""
save_drive = False
COLAB = False
if len(sys.argv) > 1:
    COLAB = True
    save_drive = True
    if len(sys.argv) > 2:
        model_name = sys.argv[1]
        gen = int(sys.argv[2])
        total_cycles = int(sys.argv[3])
        datasets = sys.argv[4:]

print(f"model_name: {model_name}", gen)

if save_drive:
    drive = DriveExplorer(on_colab=COLAB)
    drive.retrieve_model(model_name)
    for data in datasets:
        drive.retrieve_training_data(data)
    time.sleep(3)
temp = 0
while 1:
    logging.info(f"GEN: {gen}, model: {model_name}, total:{total_cycles}")
    if not temp:
        run_self_play(f"WINNER_{gen}", model_name)
    # datasets = datasets[-10:]
    new_model_name = train_net(datasets, model_name)

    res = find_best(new_model_name, model_name)
    if res[1] > res[-1] + 1:
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
