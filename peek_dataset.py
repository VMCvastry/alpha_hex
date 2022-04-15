from torch.utils.data import DataLoader

from custom_dataset import CustomDataset

test_loader_one = CustomDataset.load("./training_data", "gen1")
test_loader_one = DataLoader(test_loader_one, batch_size=1, shuffle=True)
for samples, targets1, targets2 in test_loader_one:
    print(samples.size())
    # samples.view([1, -1, 1])
    print(samples)
    # targets = targets.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    print(targets1)
    print(targets2)
    input("continue")
