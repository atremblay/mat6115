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
from mat6115.model import Net


def custom_loss(y_pred, y_true):
    y_pred, _, _ = y_pred
    loss = nn.BCEWithLogitsLoss()
    return loss(y_pred, y_true)


def acc(y_pred, y_true):
    y_pred, _, _ = y_pred
    return ((y_pred >= 0.0) == y_true).sum().float() / y_pred.shape[0]


def build_model_config(config_file):

    with open(config_file) as f:
        config = json.load(f)

    model_type = config["model_type"]
    model_config = config["model_config"]
    model_config["model_type"] = model_type
    model_config["input_size"] = config["embedding_dim"]
    model_config["padding_idx"] = TEXT.vocab.stoi[TEXT.pad_token]
    model_config["unk_idx"] = TEXT.vocab.stoi[TEXT.unk_token]
    model_config["num_embeddings"] = len(TEXT.vocab)
    model_config["embedding_dim"] = config["embedding_dim"]
    model_config["null_idx"] = TEXT.vocab["<null>"]
    model_config["dropout"] = config["dropout"]

    return model_config


def main(dataset, embedding, config_file, save_path, analyze, device):
    train_iter, test_iter = dataset_factory(dataset, embedding=embedding)

    save_path.mkdir(parents=True, exist_ok=True)
    model_config = build_model_config(config_file)
    with open(save_path / "kwargs.json", "w") as model_config_file:
        json.dump(model_config, model_config_file)

    network = Net(**model_config)

    optimizer = torch.optim.Adam(network.parameters())
    # optimizer = torch.optim.RMSprop(network.parameters())
    # optimizer = torch.optim.SGD(
    # network.parameters(), lr=0.001, nesterov=True, momentum=0.5
    # )

    writer = SummaryWriter(save_path / "runs")
    tb_logger = TensorBoardLogger(writer)

    model = Model(network, optimizer, custom_loss, batch_metrics=[acc],)
    model.to(device)

    if analysis:
        hidden = run_and_save_hidden(
            model, save_path / "flat_hidden_state_pre_training.npy", test_iter
        )
        analysis(hidden, save_path / "pca_pre_training_full.pkl")
        analysis(hidden, save_path / "pca_pre_training_2.pkl", n_components=2)

    model.fit_generator(
        train_generator=train_iter,
        valid_generator=test_iter,
        epochs=500,
        callbacks=[
            ModelCheckpoint(
                filename=str(save_path) + "/model.pkl",
                save_best_only=True,
                restore_best=True,
            ),
            tb_logger,
            EarlyStopping(min_delta=0.0001, patience=25),
        ],
    )

    with open(save_path / "full_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    if analysis:
        hidden = run_and_save_hidden(
            model, save_path / "flat_hidden_state.npy", test_iter
        )
        analysis(hidden, save_path / "pca_full.pkl")
        analysis(hidden, save_path / "pca_2.pkl", n_components=2)
