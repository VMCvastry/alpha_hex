from train_net import train_net
from self_play import run_self_play

model_name = "NET_2022-04-13_21-44-45"
model_name = "NET_2022-04-14_09-29-42"
gen = 8
while 1:
    run_self_play(f"gen{gen}", model_name)
    model_name = train_net(f"gen{gen}", model_name)
    gen += 1
    print(f"GEN: {gen}")
