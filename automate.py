from train_net import train_net
from self_play import run_self_play

model_name = ""
gen = 1
while 1:
    run_self_play("gen1", model_name)
    model_name = train_net("gen1", model_name)
    gen += 1
