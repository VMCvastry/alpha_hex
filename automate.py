from train_net import train_net
from self_play import run_self_play

model_name = ""
gen = 1
while 1:
    run_self_play(f"gen{gen}", model_name)
    model_name = train_net(f"gen{gen}", model_name)
    gen += 1
    print(f"GEN: {gen}")
