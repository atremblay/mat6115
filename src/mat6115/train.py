import torch
import os
from poutyne.framework import Model
from poutyne.framework.callbacks import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from poutyne.framework.callbacks import TensorBoardLogger
from torch import nn
from mat6115.dataset import imdb, TEXT
import json


class Net(nn.Module):

    """Basic model architecture"""

    def __init__(
        self,
        model_type,
        padding_idx,
        null_idx,
        num_embeddings,
        embedding_dim,
        dropout=0.8,
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

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.embedding.weight.data[null_idx] = torch.zeros(embedding_dim)

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
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(hidden)
        return self.predictor(hidden).squeeze(), output, output_lengths


def custom_loss(y_pred, y_true):
    y_pred, _, _ = y_pred
    loss = nn.BCEWithLogitsLoss()
    return loss(y_pred, y_true)


def get_model(config_file):
    with open(config_file) as f:
        config = json.load(f)

    model_type = config["model_type"]
    model_config = config["model_config"]
    model_config["input_size"] = config["embedding_dim"]

    model = Net(
        model_type=model_type,
        padding_idx=TEXT.vocab.stoi[TEXT.pad_token],
        null_idx=TEXT.vocab["<null>"],
        num_embeddings=len(TEXT.vocab),
        embedding_dim=config["embedding_dim"],
        **model_config
    )
    return model


def acc(y_pred, y_true):
    y_pred, _, _ = y_pred
    return ((y_pred >= 0.0) == y_true).sum().float() / y_pred.shape[0]


def main(dataset, embedding, config_file, save_path):
    if dataset == "imdb":
        train_iter, test_iter = imdb(embedding)

    save_path.mkdir(parents=True, exist_ok=True)
    network = get_model(config_file)

    optimizer = torch.optim.Adam(network.parameters(), weight_decay=0.00001)
    # optimizer = torch.optim.RMSprop(network.parameters())
    # optimizer = torch.optim.SGD(
    # network.parameters(), lr=0.001, nesterov=False, momentum=0.9
    # )

    writer = SummaryWriter(save_path / "runs")
    tb_logger = TensorBoardLogger(writer)

    model = Model(network, optimizer, custom_loss, batch_metrics=[acc],)
    model.to(torch.device("cuda", 0))
    model.fit_generator(
        train_generator=train_iter,
        valid_generator=test_iter,
        epochs=100,
        callbacks=[
            ModelCheckpoint(
                filename=str(save_path) + "/model.pkl", save_best_only=True
            ),
            tb_logger,
        ],
    )
