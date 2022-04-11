import numpy as np
import torch
from datetime import datetime
from torch import nn, optim
from matplotlib import pyplot as plt

from net import NET
from variables import *


def split_board(state):  # todo when to switch?
    first_plane = [[1 if x == 1 else 0 for x in y] for y in state]
    second_plane = [[1 if x == -1 else 0 for x in y] for y in state]
    return [first_plane, second_plane]


def process_state(state) -> torch.Tensor:
    data = torch.tensor(
        split_board(state),
        dtype=torch.float32,
    )
    data = data.unsqueeze(0)  # add batch dimension
    return data


def crap_loss(predicted_value, value, predicted_policy, policy):
    # return (value- predicted_value ) ** 2-
    # print(predicted_value, value)
    # print(predicted_policy, policy)
    return torch.mean((value - predicted_value) ** 2) + torch.mean(
        (policy - predicted_policy) ** 2
    )


class Trainer:
    def __init__(self, model=None, loss_fn=None, optimizer=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not model:
            model = NET(2, HIDDEN_FEATURES, RESNET_DEPTH, VALUE_HEAD_SIZE).to(
                self.device
            )
        self.model = model
        if not loss_fn:
            # loss_fn = nn.MSELoss(reduction="mean")
            loss_fn = crap_loss
        self.loss_fn = loss_fn
        if not optimizer:
            optimizer = optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
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
        model_path = f'models/{type(self.model).__name__}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pt'

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
            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict(), model_path)

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                # x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
                y_test = y_test.to(self.device)
                x_test = x_test.to(self.device)
                self.model.eval()
                policy, value = self.model(x_test)
                predictions.append(value.to(self.device).detach().numpy())
                values.append(y_test.to(self.device).detach().numpy())

        return predictions, values

    def poll(self, data):
        processed_data = process_state(data)
        with torch.no_grad():
            self.model.eval()  # todo check
            policy, value = self.model(processed_data)
        return policy.squeeze(0), value.squeeze(0)  # remove batch dimension
