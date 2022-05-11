from __future__ import annotations
from train_net import train_net
from self_play import run_self_play
from duel import find_best
import sys

"NET_2022-04-15_01-37-15"
model_name = "NET_2022-04-14_09-29-42"
model_name = "NET_2022-04-15_09-24-41"
gen = 21

if len(sys.argv) > 1:
    model_name = sys.argv[1]
    gen = int(sys.argv[2])
print(f"model_name: {model_name}", gen)
total_cycles = gen
datasets = ["gen8","gen23"]
# datasets = ["gen8","gen23",f"new_gen{gen}"]
temp=0
while 1:
    logging.info(f"GEN: {gen}, model: {model_name}, total:{total_cycles}")
    if not temp:
        run_self_play(f"FIXED_{gen}", model_name)
    datasets=datasets[-5:]
    new_model_name = train_net(datasets, model_name)

    res = find_best(new_model_name, model_name)
    if res[1] > res[-1]:
        model_name = new_model_name
        gen += 1
        if not temp:
            datasets.append(f"FIXED_{gen}")
    else:
        logging.info("No improvement")
    total_cycles += 1
    temp=0
