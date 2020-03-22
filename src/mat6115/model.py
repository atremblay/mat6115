import json
import os

import numpy as np
import torch
from poutyne.framework import Model
from poutyne.framework.callbacks import ModelCheckpoint, TensorBoardLogger
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from mat6115.dataset import TEXT, dataset_factory


import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout,
        pad_idx,
        rnn_type,
    ):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn_type = rnn_type.lower()
        self.rnn = []
        if self.rnn_type == "rnn":
            self.rnn = nn.ModuleList(
                [
                    nn.RNN(
                        embedding_dim if (i == 0) else hidden_dim,
                        hidden_dim,
                        dropout=dropout if (i < n_layers) else 0.0,
                        batch_first=True,
                    )
                    for i in range(n_layers)
                ]
            )
        elif self.rnn_type == "lstm":
            self.rnn = nn.ModuleList(
                [
                    nn.LSTM(
                        embedding_dim if (i == 0) else hidden_dim,
                        hidden_dim,
                        dropout=dropout if (i < n_layers) else 0.0,
                        batch_first=True,
                    )
                    for i in range(n_layers)
                ]
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.ModuleList(
                [
                    nn.GRU(
                        embedding_dim if (i == 0) else hidden_dim,
                        hidden_dim,
                        dropout=dropout if (i < n_layers) else 0.0,
                        batch_first=True,
                    )
                    for i in range(n_layers)
                ]
            )

        self.predictor = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths, hidden_state=None):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True
        )
        X = packed_embedded
        outputs = []
        for rnn in self.rnn:
            if self.rnn_type == "lstm":
                X, hidden_state = rnn(X, hidden_state)
                hidden, _ = hidden_state
            else:
                X, hidden_state = rnn(X)
                hidden = hidden_state
            # unpack sequence
            output, output_lengths = nn.utils.rnn.pad_packed_sequence(
                X, batch_first=True
            )
            outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)

        hidden = self.dropout(hidden[-1])

        return self.predictor(hidden).squeeze(-1), outputs, output_lengths, hidden_state


class Net(nn.Module):

    """Basic model architecture"""

    def __init__(
        self,
        model_type,
        padding_idx,
        null_idx,
        unk_idx,
        num_embeddings,
        embedding_dim,
        dropout=0.5,
        agg_func=None,
        **rnn_kwargs
    ):
        """

        :model_type: TODO
        :**kwargs: TODO

        """
        nn.Module.__init__(self)
        self._null_idx = null_idx
        self._embedding_dim = embedding_dim
        self._model_type = model_type
        self._num_layers = rnn_kwargs.get("num_layers", 1)
        self._agg_func = agg_func

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.embedding.weight.data[null_idx] = torch.zeros(embedding_dim)
        self.embedding.weight.data[padding_idx] = torch.zeros(embedding_dim)
        self.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)

        # self.embedding.weight[null_idx] = 0.0
        rnn_kwargs["batch_first"] = True

        if model_type == "rnn":
            self.rnn = nn.RNN(**rnn_kwargs)
        elif model_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif model_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)

        self.predictor = nn.Linear(
            in_features=rnn_kwargs["hidden_size"], out_features=1, bias=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, text_lengths, hidden_state=None):
        if X == "<null>":
            X = torch.zeros(size=(self._embedding_dim))

        embedded = self.embedding(X)
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True
        )

        if self._model_type == "lstm":
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        if self._num_layers > 1:
            if self._agg_func is None:
                # Only take the last layer hidden state
                hidden = hidden[-1]
            elif self._agg_func == "sum":
                hidden = hidden.sum(dim=0)
            elif self._agg_func == "mean":
                hidden = hidden.mean(dim=0)
            elif self._agg_func == "max":
                hidden = hidden.max(dim=0)

        hidden = self.dropout(hidden)
        return self.predictor(hidden).squeeze(), output, output_lengths
