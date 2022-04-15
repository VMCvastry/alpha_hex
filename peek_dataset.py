import torch

from torch.utils.data import DataLoader

from custom_dataset import CustomDataset
import random

# a=list(test_loader_one)
# random.shuffle(a)
s = []

test_loader_one = CustomDataset.load("./training_data", "gen1")
test_loader_one = DataLoader(test_loader_one, batch_size=1, shuffle=True)
for samples, targets1, targets2 in test_loader_one:
    # print(samples.size())
    # # samples.view([1, -1, 1])
    print(samples)
    # targets = targets.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    print(targets1)
    print(targets2)
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
