from __future__ import annotations

import random
import time

from net.train_net import train_net
from utils.self_play import run_self_play
from utils.duel import find_best
import sys
from utils.logger import logging
from utils.drive_explorer import DriveExplorer
import os

from variables import OUTPUT_LABEL

gen = 0
total_cycles = 0
datasets = [f"{OUTPUT_LABEL}_0"]
model_name = ""
save_drive = False
COLAB = False
if len(sys.argv) > 1:
    COLAB = True
    save_drive = True
    if sys.argv[1] == "clean":
        model_name = ""
        gen = 0
        total_cycles = 0
        datasets = [f"{OUTPUT_LABEL}_0"]
    elif len(sys.argv) > 2:
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
        run_self_play(f"{OUTPUT_LABEL}_{total_cycles}", model_name)
        if save_drive:
            drive.save_training_data(datasets[-1])
        datasets.append(f"{OUTPUT_LABEL}_{total_cycles}")
    if random.randint(0, 1):
        datasets = datasets[1:]
    new_model_name = train_net(datasets, model_name)

    res = find_best(new_model_name, model_name)
    if res[1] > res[-1] + 1:
        model_name = new_model_name
        if save_drive:
            drive.save_model(model_name)
        gen += 1
    else:
        logging.info("No improvement")
        if os.path.exists(f"models/{new_model_name}.pt"):
            os.remove(f"models/{new_model_name}.pt")
            logging.debug(f"deleted models/{new_model_name}.pt")
    total_cycles += 1
    temp = 0
