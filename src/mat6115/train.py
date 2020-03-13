import torch
from poutyne.framework import Model
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
        **rnn_kwargs
    ):
        """

        :model_type: TODO
        :**kwargs: TODO

        """
        nn.Module.__init__(self)
        self._null_idx = null_idx
        self._embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
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

    def forward(self, X, hidden_state=None):
        if X == "<null>":
            X = torch.zeros(size=(self._embedding_dim))

        h, k = self.rnn(self.embedding(X))
        return torch.sigmoid(self.predictor(k).squeeze(0))


def custom_loss(y_pred, y_true):
    loss = nn.BCELoss()
    return loss(y_pred, y_true)


def get_model(config_file):
    with open(config_file) as f:
        config = json.load(f)

    model_type = config["model_type"]
    model_config = config["model_config"]
    model_config["input_size"] = config["embedding_dim"]

    model = Net(
        model_type=model_type,
        padding_idx=TEXT.vocab["<pad>"],
        null_idx=TEXT.vocab["<null>"],
        num_embeddings=len(TEXT.vocab),
        embedding_dim=config["embedding_dim"],
        **model_config
    )
    return model


def main(dataset, embedding, config_file):
    if dataset == "imdb":
        train_iter, test_iter = imdb(embedding)
    print(len(train_iter))

    print(len(test_iter))
    network = get_model(config_file)

    model = Model(network, "sgd", custom_loss, batch_metrics=["accuracy"],)
    model.fit_generator(train_generator=train_iter, valid_generator=test_iter, epochs=5)
