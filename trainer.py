import time

import numpy as np
import torch
from datetime import datetime
from torch import nn, optim
from matplotlib import pyplot as plt

from net import NET
from variables import *


def split_board(state, player):  # todo when to switch?
    current_player_plane = [[1 if x == player else 0 for x in y] for y in state]
    second_plane = [[1 if x == -1 * player else 0 for x in y] for y in state]
    return [current_player_plane, second_plane]


def process_state(state, player: int) -> torch.Tensor:
    data = torch.tensor(
        split_board(state, player),
        dtype=torch.float32,
    )
    return data


def crap_loss(predicted_value, value, predicted_policy, policy):
    # return (value- predicted_value ) ** 2-
    # print(predicted_value, value)
    # print(predicted_policy, policy)
    return torch.mean((value - predicted_value) ** 2) + torch.mean(
        (policy - predicted_policy) ** 2
    )


class Trainer:
    def __init__(
        self, *args, model_name=None, model=None, loss_fn=None, optimizer=None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not model:
            model = NET(2, HIDDEN_FEATURES, RESNET_DEPTH, VALUE_HEAD_SIZE).to(
                self.device
            )
            if model_name:
                model.load_state_dict(torch.load(f"models/{model_name}.pt"))
                model.eval()
        self.model = model
        if not loss_fn:
            # loss_fn = nn.MSELoss(reduction="mean")
            loss_fn = crap_loss
        self.loss_fn = loss_fn
        if not optimizer:
            optimizer = optim.SGD(
                model.parameters(), momentum=MOMENTUM, lr=LEARNING_RATE
            )
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, state, value, policy):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        predicted_policy, predicted_value = self.model(state)

        # # Computes loss
        # value_loss = self.loss_fn(value, predicted_value)
        # policy_loss = self.loss_fn(policy, predicted_policy)
        # # Computes gradients
        # value_loss.backward()
        # policy_loss.backward()
        loss = self.loss_fn(predicted_value, value, predicted_policy, policy)
        loss.backward()
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        model_name = f'{type(self.model).__name__}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, value_batch, policy_batch in train_loader:
                # x_batch = x_batch.view([batch_size, -1, n_features]).to(self.device)
                value_batch = value_batch.to(self.device)
                policy_batch = policy_batch.to(self.device)
                loss = self.train_step(x_batch, value_batch, policy_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            # with torch.no_grad():
            #     batch_val_losses = []
            #     for x_val, y_val in val_loader:
            #         # x_val = x_val.view([batch_size, -1, n_features]).to(self.device)
            #         y_val = y_val.to(self.device)
            #         self.model.eval()
            #         policy, value = self.model(x_val)
            #         val_loss = self.loss_fn(y_val, value).item()
            #         batch_val_losses.append(val_loss)
            #     validation_loss = np.mean(batch_val_losses)
            #     self.val_losses.append(validation_loss)
            validation_loss = -1
            if (epoch <= 10) | (epoch % 50 == 0) | (epoch == n_epochs):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f},{datetime.now()}"
                )

        torch.save(self.model.state_dict(), f"models/{model_name}.pt")
        return model_name

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            values_predictions = []
            policies_predictions = []
            values = []
            policies = []
            for x_test, value, policy in test_loader:
                # x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
                value = value.to(self.device)
                policy = policy.to(self.device)
                x_test = x_test.to(self.device)
                self.model.eval()
                predicted_policy, predicted_value = self.model(x_test)
                values_predictions.append(
                    predicted_value.to(self.device).detach().numpy()
                )
                policies_predictions.append(
                    predicted_policy.to(self.device).detach().numpy()
                )
                values.append(value.to(self.device).detach().numpy())
                policies.append(policy.to(self.device).detach().numpy())

        return values_predictions, values, policies_predictions, policies

    def poll(self, data, player):
        processed_data = process_state(data, player)
        processed_data = processed_data.unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            self.model.eval()  # todo check
            policy, value = self.model(processed_data)
        return policy.squeeze(0), value.squeeze(0)  # remove batch dimension
