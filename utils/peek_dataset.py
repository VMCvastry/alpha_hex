from __future__ import annotations
import torch

from torch.utils.data import DataLoader

from utils.custom_dataset import CustomDataset
from game import Game, discard_rows_for_print

# a=list(test_loader_one)
# random.shuffle(a)
s = []
test_loader_one = CustomDataset.load("./training_data", "R_1_HEX_8")
test_loader_one = CustomDataset.load("./training_data", "R_2_HEX_17")
test_loader_one = CustomDataset.load("./training_data", "R_3_HEX_10")
test_loader_one = CustomDataset.load("./training_data", "R_3_HEX_23")
test_loader_one = CustomDataset.load("./training_data", "R_3_HEX_108")
test_loader_one = CustomDataset.load("./training_data", "R_3_HEX_150")
test_loader_one = CustomDataset.load("./training_data", "R_3_HEX_181")
test_loader_one = DataLoader(test_loader_one, batch_size=1, shuffle=True)
for samples, targets1, targets2 in test_loader_one:
    # print(samples.size())
    # # samples.view([1, -1, 1])

    if torch.sum(samples) > 10:
        print(samples)
        pretty_state = samples[0][0] + -1 * samples[0][1]
        g = Game(pretty_state.type(torch.IntTensor).tolist())
        print(g)
        print(g.winner)
        print(torch.sum(samples))
        # targets = targets.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        print(targets1)
        [
            print(list(map("{:.2f}".format, x)))
            for x in discard_rows_for_print(targets2.tolist()[0])
        ]
        input("continue")
#     s.append(samples)
# print(len(s))
#
# unique = []
# for ss in s:
#     flag = True
#     for sss in unique:
#         if torch.equal(ss, sss):
#             flag = False
#             break
#     if flag:
#         unique.append(ss)
# print(len(unique))
