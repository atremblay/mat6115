from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from mat6115.dataset import dataset_factory, TEXT, LABEL, SEED
from mat6115.model import RNN
from mat6115.train import custom_loss, acc
from mat6115.fixed_point import FixedPointFinder
from poutyne.framework import Model
import torch
from torch import nn
from pathlib import Path
import pickle
import json

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def load_model(save_path, restore=False):
    network = RNN(**kwargs)

    model = Model(
        network=network,
        optimizer=torch.optim.Adam(network.parameters()),
        loss_function=custom_loss,
        batch_metrics=[acc],
    )
    if restore:
        model.load_weights(SAVE_PATH / "model.pkl")
    return model


def get_hidden(model, iterator, N=1000):
    batch_size = iterator.batch_size
    num_batch = N // batch_size

    _, _, preds, y_true = model.evaluate_generator(
        iterator, return_pred=True, return_ground_truth=True, steps=num_batch + 1
    )
    # y_pred, outputs, output_lenghts = zip(*preds)

    flat_hidden_state = []
    num_hidden_states = 0
    ground_truth = []
    model_preds = []
    for i, (y_pred, outputs, output_lengths, _) in enumerate(preds):
        for j, (output, length) in enumerate(zip(outputs, output_lengths)):
            num_hidden_states += length
            flat_hidden_state.append([o[:length] for o in output])
            ground_truth.extend([y_true[i][j]] * length)
            model_preds.extend([int(y_pred[j] > 0.0)] * length)
        if i >= num_batch:
            break

    num_layers, _, hidden_state_dim = output.shape
    # fhs = np.empty((num_hidden_states, num_layers, hidden_state_dim))
    fhs = []
    for _ in range(num_layers):
        fhs.append([])

    for batch in flat_hidden_state:
        for i, output in enumerate(batch):
            fhs[i].extend(output)

    flat_hidden_state = np.array(fhs)
    ground_truth = np.array(ground_truth)
    model_preds = np.array(model_preds)
    assert flat_hidden_state.shape[1] == num_hidden_states
    return flat_hidden_state, model_preds, ground_truth


if __name__ == "__main__":
    train_iter, valid_iter, test_iter = dataset_factory(
        "imdb", embedding="glove.6B.100d"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    SAVE_PATH = Path("gru_1layer")
    with open(SAVE_PATH / "kwargs.json", "r") as kwargs_file:
        kwargs = json.load(kwargs_file)

    num_layers = kwargs["n_layers"]

    vanilla_model = load_model(SAVE_PATH)
    trained_model = load_model(SAVE_PATH, restore=True)

    vanilla_model.to(device)
    trained_model.to(device)

    trained_hidden_states, trained_preds, trained_ground_truth = get_hidden(
        trained_model, test_iter, N=5000
    )

    fixed_point_finder = FixedPointFinder(
        rnn_cell=trained_model.network.rnn[0],
        lr=0.01,
        n_iter=50000,
        device=device,
        batch_size=512,
    )

    trained_hidden_states = torch.tensor(trained_hidden_states[:, :5000]).to(device)
    constant_input = torch.zeros(
        (trained_hidden_states.shape[1], 1, kwargs["embedding_dim"])
    ).to(device)

    point, diff = fixed_point_finder.run(trained_hidden_states, constant_input)
    is_fixed_point = diff < 1e-8
    print(f"Found {is_fixed_point.sum()} fixed points")
    print(f"min q-value: {diff.min()}")
    __import__("pudb").set_trace()
    np.save(open(SAVE_PATH / "fixed_points.npy", "wb"), point[is_fixed_point])
