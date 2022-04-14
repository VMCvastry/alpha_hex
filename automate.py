from train_net import train_net
from self_play import run_self_play
from duel import find_best

model_name = "NET_2022-04-14_09-29-42"
gen = 8
while 1:
    print(f"GEN: {gen}")
    run_self_play(f"gen{gen}", model_name)
    new_model_name = train_net(f"gen{gen}", model_name)
    gen += 1
    res = find_best(new_model_name, model_name)
    if res[1] >= res[-1]:
        model_name = new_model_name
    else:
        print("No improvement")
