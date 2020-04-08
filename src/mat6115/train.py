import json
import os
import pickle

import torch
from poutyne.framework import Model
from poutyne.framework.callbacks import (
    ModelCheckpoint,
    TensorBoardLogger,
    EarlyStopping,
)
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from mat6115.dataset import TEXT, dataset_factory
from mat6115.hidden import run_and_save_hidden, analysis
from mat6115.model import RNN
from pathlib import Path


def custom_loss(y_pred, y_true):
    y_pred = y_pred[0]
    loss = nn.BCEWithLogitsLoss()
    return loss(y_pred, y_true)


def acc(y_pred, y_true):
    y_pred = y_pred[0]
    return ((y_pred >= 0.0) == y_true).sum().float() / y_pred.shape[0]


def main(rnn_type, n_layers, dataset, embedding, device, save_path):
    train_iter, valid_iter, test_iter = dataset_factory(dataset, embedding=embedding)
    embedding_dim = int(embedding.split(".")[-1][:-1])
    save_path = Path(save_path) / f"{rnn_type}_{n_layers}layer_{embedding_dim}"
    save_path.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        vocab_size=len(TEXT.vocab),
        embedding_dim=embedding_dim,
        hidden_dim=256,
        output_dim=1,
        n_layers=n_layers,
        dropout=0.5,
        pad_idx=TEXT.vocab.stoi[TEXT.pad_token],
        rnn_type="gru",
    )
    with open(save_path / "kwargs.json", "w") as kwargs_file:
        json.dump(kwargs, kwargs_file)

    pretrained_embeddings = TEXT.vocab.vectors

    network = RNN(**kwargs)
    network.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    network.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
    network.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)

    optimizer = torch.optim.Adam(network.parameters())
    model = Model(
        network=network,
        optimizer=optimizer,
        loss_function=custom_loss,
        batch_metrics=[acc],
    )
    model.to(device)

    history = model.fit_generator(
        train_generator=train_iter,
        valid_generator=valid_iter,
        epochs=10,
        callbacks=[
            ModelCheckpoint(
                filename=str(save_path / "model.pkl"),
                save_best_only=True,
                restore_best=True,
            )
        ],
    )
    print(f"Model saved to {save_path}")
    __import__("pudb").set_trace()
    test_loss, test_acc, y_pred, y_true = model.evaluate_generator(
        generator=test_iter, return_pred=True, return_ground_truth=True
    )
    print(f"Test Loss: {test_loss:.4f}, Test Binary Accuracy: {test_acc:.4f}")
