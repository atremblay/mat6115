from poutyne.framework import Model
from poutyne.framework.callbacks import EarlyStopping
import torch
from torch import nn
from typing import Union, Tuple
import numpy as np


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
    if isinstance(y_pred, tuple):
        y_pred, _ = y_pred
    return torch.dist(y_true, y_pred) ** 2 / y_true.shape[-1]


class RNNWrapper(nn.Module):

    """
    Wrapper around RNN to support giving the hidden state as starting point
    """

    def __init__(self, rnn_cell):
        """


        :rnn_cell: TODO

        """
        nn.Module.__init__(self)

        self._rnn_cell = rnn_cell

    def forward(self, constant_input, hidden_state):
        __import__("pudb").set_trace()
        return self._rnn_cell(constant_input, hidden_state)


class FixedPointFinder(object):

    """Utility class to find fixed point of a Pytorch RNN Cell"""

    def __init__(self, rnn_cell, lr=0.001, n_iter=200000):
        """
        Parameters
        ----------
        rnn_cell: torch.nn.RNNBase
            Any Pytorch layer that conform to RNNBase

        lr: float
            Learning Rate. Using SGD optimizer.
        """
        self._rnn_cell = RNNWrapper(rnn_cell)
        self._rnn_cell = rnn_cell
        self._rnn_cell.to(device)
        self._lr = lr
        self._n_iter = n_iter

        for param in self._rnn_cell.parameters():
            param.requires_grad = False

    def run(self, hidden_state, constant_input):
        constant_input.requires_grad = False
        initial = hidden_state.clone()

        hidden_state.requires_grad = True

        model = Model(
            network=self._rnn_cell,
            loss_function=speed_loss,
            optimizer=torch.optim.SGD(params=[hidden_state], lr=self._lr),
        )
        # model.to(torch.device("cuda", 0))
        model.fit(
            (constant_input, hidden_state),
            hidden_state,
            batch_size=32,
            epochs=self._n_iter,
            callbacks=[EarlyStopping(monitor="loss", min_delta=1e-8)],
            verbose=True,
        )
        return hidden_state.squeeze(), (speed_loss(initial, hidden_state) < 1e-8)


if __name__ == "__main__":
    device = torch.device("cuda", 0)
    rnn = nn.GRU(input_size=100, hidden_size=256)
    constant_input = torch.zeros((1, 2 ** 16, 100)).to(device)
    hidden_state = nn.init.normal_(torch.empty((1, 2 ** 16, 256))).to(device)

    fixed_point_finder = FixedPointFinder(rnn_cell=rnn, n_iter=2000)
    print(fixed_point_finder.run(hidden_state, constant_input)[1])


