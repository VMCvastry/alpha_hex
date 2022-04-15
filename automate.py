from train_net import train_net
from self_play import run_self_play
from duel import find_best

"NET_2022-04-15_01-37-15"
model_name = "NET_2022-04-14_09-29-42"
model_name = "NET_2022-04-15_09-24-41"
gen = 21
total_cycles=gen
while 1:
    print(f"GEN: {gen}, model: {model_name}, total:{total_cycles}")
    run_self_play(f"gen{gen}", model_name)
    new_model_name = train_net(f"gen{gen}", model_name)

    res = find_best(new_model_name, model_name)
    if res[1] >= res[-1]:
        model_name = new_model_name
        gen += 1
    else:
        print("No improvement")
    total_cycles+=1
