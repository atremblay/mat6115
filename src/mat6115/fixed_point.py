from poutyne.framework import Model
from poutyne.framework.callbacks import EarlyStopping, ModelCheckpoint
from poutyne.framework.callbacks.lr_scheduler import StepLR
import torch
from tempfile import NamedTemporaryFile
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
    if isinstance(y_pred, tuple):
        y_pred, _ = y_pred
    return ((y_true - y_pred) ** 2).mean()


class RNNWrapper(nn.Module):

    """
    Wrapper around a single RNNCell or a list of RNNCells.

    If a single RNNCell is given, then it behaves exactly like the original
    RNN cell. If it's a list, it iterates through the list, passing the
    intermediary hidden states as input to the next layer.

    The goal of this is to circumvent the limitation of multiple layers
    of Pytorch *not* returning the intermediate layers.
    """

    def __init__(self, rnn_cell):
        """TODO: to be defined.

        :rnn_cell: TODO

        """
        nn.Module.__init__(self)

        self._rnn_cell = rnn_cell

    def forward(self, constant_input, hidden_state):
        return self._rnn_cell(constant_input, hidden_state)


class FixedPointFinder(object):

    """Utility class to find fixed point of a Pytorch RNN Cell"""

    def __init__(
        self,
        rnn_cell,
        device=torch.device("cpu"),
        lr=0.001,
        n_iter=2000,
        batch_size=512,
    ):
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
        self._rnn_cell.to(device)
        self._lr = lr
        self._device = device
        self._n_iter = n_iter

    def run(self, constant_input, hidden_state):
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
                num_items = self.constant_input.shape[0]
                l = num_items // self._batch_size
                l += 1 if num_items % self._batch_size != 0 else 0
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
            optimizer=torch.optim.Adam(params=[hidden_state], lr=self._lr),
        )

        model.fit_generator(
            DataIterator(constant_input, hidden_state, batch_size=self._batch_size),
            epochs=self._n_iter,
            verbose=False,
            callbacks=[
                StepLR(step_size=1000, gamma=0.5),
                EarlyStopping(monitor="loss", min_delta=1e-6, patience=1000),
                ModelCheckpoint(
                    filename=NamedTemporaryFile().name,
                    monitor="loss",
                    save_best_only=True,
                    restore_best=True,
                ),
            ],
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

    def calc_jacobian(self, fixed_point, constant_input):
        fixed_point = fixed_point.to(self._device)
        fixed_point.requires_grad = True
        constant_input = constant_input.to(self._device)
        constant_input.requires_grad = True
        output, _ = self._rnn_cell(constant_input, fixed_point)

        jacobian_h = torch.zeros(self._rnn_cell.hidden_size, self._rnn_cell.hidden_size)
        jacobian_i = torch.zeros(self._rnn_cell.hidden_size, self._rnn_cell.input_size)
        for i in range(self._rnn_cell.hidden_size):
            grad_output = torch.zeros(1, 1, self._rnn_cell.hidden_size).to(self._device)
            grad_output[0, 0, i] = 1.0
            # Other option would be to use torch.autograd.backward
            # Then we would have to `.grad.zero_()` the fixed_point
            # at each iteration
            jacobians = torch.autograd.grad(
                output,
                (fixed_point, constant_input),
                grad_outputs=grad_output,
                retain_graph=True,
            )
            jacobian_h[i] = jacobians[0].squeeze()
            jacobian_i[i] = jacobians[1].squeeze()

        return jacobian_h.numpy(), jacobian_i.numpy()

