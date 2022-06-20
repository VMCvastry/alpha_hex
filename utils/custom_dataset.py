from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset
from pathlib import Path
import dill
from utils.logger import logging
from variables import GRID_SIZE


class CustomDataset(Dataset):
    def __init__(self, states, values, policies):
        super().__init__()
        self.states = states
        self.values = values
        self.policies = policies
        assert self.states.size()[0] == self.values.size()[0] == self.policies.size()[0]

    def __len__(self):
        return self.states.size()[0]

    def append(self, states, values, policies):
        self.states = torch.cat((self.states, states), dim=0)
        self.values = torch.cat((self.values, values), dim=0)
        self.policies = torch.cat((self.policies, policies), dim=0)
        assert self.states.size()[0] == self.values.size()[0] == self.policies.size()[0]

    def remove_duplicates(self):
        _, indexes = torch.unique(self.states, dim=0, return_inverse=True)
        new_states = torch.zeros(_.size()[0], 2, GRID_SIZE, GRID_SIZE)
        new_values = torch.zeros(_.size()[0], 1, 1, 1)
        new_policies = torch.zeros(_.size()[0], GRID_SIZE, GRID_SIZE)
        for i, index in enumerate(indexes):
            new_states[index] = self.states[i]
            new_values[index] = self.values[i]
            new_policies[index] = self.policies[i]
        logging.info(
            f"removed duplicate states, prev:{self.states.size()[0]}, now: {new_states.size()[0]}"
        )
        self.states, self.values, self.policies = new_states, new_values, new_policies

    def append_dataset(self, dataset: CustomDataset):
        self.states = torch.cat((self.states, dataset.states), dim=0)
        self.values = torch.cat((self.values, dataset.values), dim=0)
        self.policies = torch.cat((self.policies, dataset.policies), dim=0)
        self.remove_duplicates()  # todo check removing older
        assert self.states.size()[0] == self.values.size()[0] == self.policies.size()[0]

    def __getitem__(self, idx):
        # return {
        #     "state": self.states[idx],
        #     "value": self.values[idx],
        #     "policy": self.policies[idx],
        # }
        return (
            self.states[idx],
            self.values[idx],
            self.policies[idx],
        )

    def store(self, path, name):
        self.remove_duplicates()
        torch.save(self.states, f"{path}/{name}_states.pkl", pickle_module=dill)
        torch.save(self.values, f"{path}/{name}_values.pkl", pickle_module=dill)
        torch.save(self.policies, f"{path}/{name}_policies.pkl", pickle_module=dill)

    @classmethod
    def load(cls, path, name):

        states = torch.load(f"{path}/{name}_states.pkl", pickle_module=dill)
        values = torch.load(f"{path}/{name}_values.pkl", pickle_module=dill)
        policies = torch.load(f"{path}/{name}_policies.pkl", pickle_module=dill)
        return cls(states, values, policies)

    # def remove_zero_value(self):
    #     mask = self.values != 0
    #     mask = mask.squeeze()
    #     self.states = self.states[mask]
    #     self.values = self.values[mask]
    #     self.policies = self.policies[mask]
    #
    # def remove_ones_value(self):
    #     mask = self.values != 1
    #     mask = mask.squeeze()
    #     self.states = self.states[mask]
    #     self.values = self.values[mask]
    #     self.policies = self.policies[mask]
    #
    # def even(self):
    #     rem = 0
    #     while rem < 104:
    #         i = random.randint(0, len(self.states) - 1)
    #         if self.values[i] == 1:
    #             self.states = torch.cat((self.states[:i], self.states[i + 1 :]), 0)
    #             self.values = torch.cat((self.values[:i], self.values[i + 1 :]), 0)
    #             self.policies = torch.cat(
    #                 (self.policies[:i], self.policies[i + 1 :]), 0
    #             )
    #             rem += 1
