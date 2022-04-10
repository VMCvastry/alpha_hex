import numpy as np
import torch
from datetime import datetime

from matplotlib import pyplot as plt


def split_board(state):  # todo when to switch?
    first_plane = [[1 if x == 1 else 0 for x in y] for y in state]
    second_plane = [[1 if x == -1 else 0 for x in y] for y in state]
    return [first_plane, second_plane]


def process_state(state) -> torch.Tensor:
    data = torch.tensor(
        split_board(state),
        dtype=torch.float32,
    )  # todo add batch dimension
    return data


class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
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
            for x_batch, y_batch in train_loader:
                # x_batch = x_batch.view([batch_size, -1, n_features]).to(self.device)
                y_batch = y_batch.to(self.device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    # x_val = x_val.view([batch_size, -1, n_features]).to(self.device)
                    y_val = y_val.to(self.device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

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
        return policy, value
