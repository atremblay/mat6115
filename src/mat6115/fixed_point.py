from poutyne.framework import Model
from poutyne.framework.callbacks import EarlyStopping
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from typing import Union, Tuple
import numpy as np


def _speed_loss(y_pred, y_true):
    if isinstance(y_pred, tuple):
        y_pred, _ = y_pred
    return ((y_pred - y_true) ** 2).squeeze().sum(axis=-1) / y_true.shape[-1]


def speed_loss(
    y_pred: Tuple[np.ndarray, Union[Tuple[np.ndarray, np.ndarray], np.ndarray]],
    y_true: np.ndarray,
):
    """Loss function to use for fixed point finder

    y_pred: Tuple[np.ndarray, Union[Tuple[np.ndarray, np.ndarray], np.ndarray]]
        The output of the RNN Cell. This will take care of LSTM's special case
        of returning the hidden and cell state.

    y_true: np.ndarray
        This will be the same as the input.
        TODO: Should this be a copy and put requires_grad = False?

    """
    # Using the fact that the output of any RNN cell is the last hidden state
    # In the case of a fixed point we pass only one input, so this corresponds
    # to the hidden state we are interested in.
    # return _speed_loss(y_pred, y_true).sum() / np.prod(y_true.shape)
    y_pred, _ = y_pred
    return torch.dist(y_true, y_pred) ** 2 / y_true.shape[-2]  # np.prod(y_true.shape)


class RNNWrapper(nn.Module):

    """Docstring for RNNWrapper. """

    def __init__(self, rnn_cell):
        """TODO: to be defined.

        :rnn_cell: TODO

        """
        nn.Module.__init__(self)

        self._rnn_cell = rnn_cell

    def forward(self, constant_input, hidden_state):
        return self._rnn_cell(constant_input, hidden_state)


def collate_fn(data):
    constant_input = torch.cat([d[0][0] for d in data], dim=0).unsqueeze(1)
    hidden_state = torch.cat([d[0][1] for d in data], dim=0).unsqueeze(0)
    target = torch.cat([d[1] for d in data], dim=0).unsqueeze(0)
    return (constant_input, hidden_state), target


class FixedPointFinder(object):

    """Utility class to find fixed point of a Pytorch RNN Cell"""

    def __init__(self, rnn_cell, device, lr=0.001, n_iter=200000, batch_size=512):
        """
        Parameters
        ----------
        rnn_cell: torch.nn.RNNBase
            Any Pytorch layer that conform to RNNBase

        lr: float
            Learning Rate. Using SGD optimizer.
        """
        self._rnn_cell = rnn_cell
        self._batch_size = batch_size
        # self._rnn_cell = RNNWrapper(rnn_cell)
        self._rnn_cell.to(device)
        self._lr = lr
        self._device = device
        self._n_iter = n_iter

        for param in self._rnn_cell.parameters():
            param.requires_grad = False

    def run(self, hidden_state, constant_input):
        constant_input.requires_grad = False
        hidden_state.requires_grad = True

        class DataIterator(Dataset):

            """
            Required dataset class to circumvent some of poutyne's limitation.
            The short version is that calling `.fit()` creates a TensorDataset
            (poutyne's own version) and it checks that the first dimension of all
            inputs are of same dimensions. The nature of RNNCell makes it such
            that the input and the hidden state cannot be aligned on the first
            dimension.

            So we create our own iterator and use `.fit_generator()` instead
            """

            def __init__(self, constant_input, hidden_state, batch_size=32):
                self.constant_input = constant_input
                self.hidden_state = hidden_state
                self._batch_size = batch_size
                assert self.constant_input.shape[0] == self.hidden_state.shape[1]

            def __len__(self):
                l = self.constant_input.shape[0] // self._batch_size
                l += 1 if l % self._batch_size != 0 else 0
                return l

            def __iter__(self):
                last_idx = self.constant_input.shape[0]
                last_idx += last_idx % self._batch_size
                for start in range(0, last_idx, self._batch_size):
                    end = start + self._batch_size
                    x = self.constant_input[start:end]
                    y = self.hidden_state[:, start:end]
                    yield (x, y), y

        model = Model(
            network=self._rnn_cell,
            loss_function=speed_loss,
            optimizer=torch.optim.SGD(params=[hidden_state], lr=self._lr),
        )

        model.fit_generator(
            DataIterator(constant_input, hidden_state, batch_size=self._batch_size),
            epochs=self._n_iter,
            verbose=True,
        )

        trained = hidden_state.clone().detach()
        _, output = model.evaluate_generator(
            DataIterator(constant_input, trained, batch_size=self._batch_size),
            return_pred=True,
        )

        output = np.concatenate([o[0] for o in output])

        if trained.device.type == "cuda":
            trained = trained.detach().cpu().numpy()
            hidden_state = hidden_state.detach().cpu().numpy()
        else:
            trained = trained.detach().numpy()
            hidden_state = hidden_state.detach().numpy()
        return (
            hidden_state.squeeze(),
            _speed_loss(np.squeeze(trained), np.squeeze(output)),
        )


if __name__ == "__main__":
    device = torch.device("cuda", 0)
    rnn = nn.GRU(input_size=100, hidden_size=256, batch_first=True)
    constant_input = torch.zeros((2 ** 16, 1, 100)).to(device)
    hidden_state = nn.init.normal_(torch.empty((1, 2 ** 16, 256))).to(device)

    fixed_point_finder = FixedPointFinder(
        rnn_cell=rnn, device=device, n_iter=20000, lr=0.01
    )
    point, is_fixed_point = fixed_point_finder.run(
        hidden_state, constant_input, batch_size=256
    )
    print(f"Found {is_fixed_point.sum()} fixed points")

