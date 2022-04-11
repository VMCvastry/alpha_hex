import torch
from torch.utils.data import Dataset
from pathlib import Path
import dill


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
        torch.save(self.states, f"{path}/{name}_states.pkl", pickle_module=dill)
        torch.save(self.values, f"{path}/{name}_values.pkl", pickle_module=dill)
        torch.save(self.policies, f"{path}/{name}_policies.pkl", pickle_module=dill)

    @classmethod
    def load(cls, path, name):

        states = torch.load(f"{path}/{name}_states.pkl", pickle_module=dill)
        values = torch.load(f"{path}/{name}_values.pkl", pickle_module=dill)
        policies = torch.load(f"{path}/{name}_policies.pkl", pickle_module=dill)
        return cls(states, values, policies)
