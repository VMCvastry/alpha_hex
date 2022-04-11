from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from custom_dataset import CustomDataset
from net import NET
from variables import *
from trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train = torch.load("data/train_dataset.pt")
# val = torch.load("data/val_dataset.pt")
# test = torch.load("data/test_dataset.pt")
# train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
# val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
# test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
# test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
# data = torch.tensor(
#     [[[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0], [1, 0, 0]]]
# )
test_loader_one = torch.tensor(
    [[[[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0], [1, 0, 0]]]],
    dtype=torch.float32,
)

test_loader_one = torch.tensor(
    [[[[1, 1, 1]] * 3] * 2] * N + [[[[0, 0, 0]] * 3] * 2] * N,
    dtype=torch.float32,
)
test = torch.tensor(
    [[[[1, 1, 1]] * 3] * 2] * 3 + [[[[0, 0, 0]] * 3] * 2] * 3,
    dtype=torch.float32,
)
dataset = CustomDataset(
    test_loader_one,
    torch.tensor([1] * N + [0] * N, dtype=torch.float32)
    .unsqueeze(1)
    .unsqueeze(1)
    .unsqueeze(1),
    torch.tensor(
        [[[1, 1, 1]] * 3] * N + [[[0, 0, 0]] * 3] * N,
        dtype=torch.float32,
    ),
)

datasetT = TensorDataset(
    test,
    torch.tensor([1] * 3 + [0] * 3, dtype=torch.float32)
    .unsqueeze(1)
    .unsqueeze(1)
    .unsqueeze(1),
)
test_loader_one = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
)
test = DataLoader(datasetT, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
for samples, targets1, targets2 in test_loader_one:
    print(samples.size())
    # samples.view([1, -1, 1])
    print(samples)
    # targets = targets.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    print(targets1)
    print(targets2)
    exit(1)


# for images, labels in test_loader_one:
#     images = images.to(device)
#
#     labels = labels.to(device)
#     outputs = model(images)
#     print(outputs)
#     print(outputs.size())
# exit(1)
trainer = Trainer()
trainer.train(
    test_loader_one,
    val_loader=[],
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    n_features=2,
)
# opt.plot_losses()
predictions, values = trainer.evaluate(test, batch_size=1, n_features=2)
print(predictions, values)
