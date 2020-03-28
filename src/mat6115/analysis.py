from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from mat6115.dataset import dataset_factory, TEXT, LABEL, SEED
from mat6115.model import RNN
from mat6115.train import custom_loss, acc
from mat6115.fixed_point import FixedPointFinder, RNNWrapper
from poutyne.framework import Model
import tqdm
import torch
from torch import nn
from pathlib import Path
import pickle
import json

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def load_model(load_path, restore=False):
    load_path = Path(load_path)
    with open(load_path / "kwargs.json", "r") as kwargs_file:
        kwargs = json.load(kwargs_file)

    network = RNN(**kwargs)

    model = Model(
        network=network,
        optimizer=torch.optim.Adam(network.parameters()),
        loss_function=custom_loss,
        batch_metrics=[acc],
    )
    if restore:
        model.load_weights(load_path / "model.pkl")
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


def _unique_fixed_from_disk(load_path, threshold=1e-3):
    with open(load_path / "fixed_points.npy", "rb") as f:
        fixed_points = np.load(f)
    return unique_fixed(fixed_points, threshold)


def unique_fixed(fixed_points, threshold=1e-3):
    """
    Remove all duplicated (fixed) points. Using euclidean distance
    less than `threshold` to determine if they are the same.

    Parameters
    ----------
    fixed_points: np.ndarray
        Shape (N, D) where N is the number of fixed points and D the
        number of dimensions.

    threshold: float (default 1e-8)
        If the euclidean distance is less than this value then they
        are considered the same fixed point.

    Returns
    -------
    np.ndarray
        Shape (M, D) where M is the number of unique fixed point.
        Should be returned in the same order they were received,
        but there is no guarantee for this (I did not test it).
    """
    if len(fixed_points) == 0:
        print("No fixed point provided")
        return np.array([])

    unique = []
    queue = [0]
    while len(queue) > 0:
        current = queue.pop(0)
        unique.append(current)
        for idx in range(current + 1, len(fixed_points)):
            # if np.linalg.norm(fixed_points[current] - fixed_points[idx]) > threshold:
            if ((fixed_points[current] - fixed_points[idx]) > threshold).any():
                if idx in queue or idx in unique:
                    continue
                queue.append(idx)
            elif idx in queue:
                queue.pop(queue.index(idx))
    return fixed_points[np.array(unique)]


def find_fixed_point(rnn, constant_input, hidden_states, device):
    points, diffs = [], []
    for i in tqdm.trange(constant_input.shape[0]):
        fixed_point_finder = FixedPointFinder(
            rnn_cell=rnn, lr=0.01, n_iter=50000, device=device, batch_size=512,
        )
        point, diff = fixed_point_finder.run(
            constant_input[i : i + 1], hidden_states[:, i : i + 1]
        )
        points.append(point)
        diffs.append(diff)
    return np.array(points), np.array(diffs)


def main(
    load_path,
    dataset,
    embedding,
    rnn_layer,
    device,
    fixed_point=False,
    unique_fixed_point=False,
    pca=False,
):
    """
    Parameters
    ----------
    load_path: Path
        Path to the model

    dataset: str
        Dataset to use. Only support IMDB at the moment.

    embedding: str
        What embedding was used during training.
        (should probably save this as part of the training)

    rnn_layer: int
        1, 2, ... Create artifacts for that rnn layer

    fixed_point: bool (default False)
        Run the fixed point finder for the specified rnn layer with random
        5000 hidden states from the test set. This can take a while.

    unique_fixed_point: bool (default False)
        Find unique fixed point. Typically many of those found by the fixed point
        finder are not unique. Typically you want to run this at the same time.

    pca: bool (default False)
        Run PCA on untrained and trained model and PCA(n_components=2) on
        trained model. Again, done on the specified rnn layer. Done with
        5000 random hidden state of the test set.
    """
    if not fixed_point and not unique_fixed_point and not pca:
        return

    save_path = load_path / str(rnn_layer)
    save_path.mkdir(parents=True, exist_ok=True)
    rnn_layer -= 1
    if fixed_point or pca:
        train_iter, valid_iter, test_iter = dataset_factory(
            dataset, embedding=embedding
        )
        with open(load_path / "kwargs.json", "r") as kwargs_file:
            kwargs = json.load(kwargs_file)

        num_layers = kwargs["n_layers"]

        trained_model = load_model(load_path, restore=True)

        trained_model.to(device)
        trained_hidden_states, _, trained_ground_truth = get_hidden(
            trained_model, test_iter, N=1000
        )

    if fixed_point:
        random_idx = np.arange(trained_hidden_states.shape[1])
        np.random.shuffle(random_idx)
        random_idx = random_idx[:1000]
        # trained_hidden_states = trained_hidden_states[:, random_idx]
        hidden_states = (
            torch.tensor(trained_hidden_states[rnn_layer, random_idx])
            .unsqueeze(0)
            .to(device)
        )

        input_dim = trained_model.network.rnn[rnn_layer].input_size
        num_inputs = hidden_states.shape[1]
        constant_input = torch.zeros((num_inputs, 1, input_dim)).to(device)

        point, diff = find_fixed_point(
            trained_model.network.rnn[rnn_layer], constant_input, hidden_states, device
        )
        is_fixed_point = diff <= 1e-6
        print(f"{is_fixed_point.sum()} points converged.")
        print(f"min q-value: {diff.min()}")
        np.save(
            open(save_path / "fixed_points.npy", "wb"), point[is_fixed_point],
        )
        np.save(
            open(save_path / "converged_points.npy", "wb"), point,
        )
        np.save(
            open(save_path / "diff.npy", "wb"), diff,
        )

    if unique_fixed_point:
        unique = _unique_fixed_from_disk(save_path)
        np.save(open(save_path / "unique_fixed_points.npy", "wb"), unique)
        print(f"Found {len(unique)} unique fixed points")

    if pca:
        vanilla_model = load_model(load_path)
        vanilla_model.to(device)
        vanilla_hidden_states, _, vanilla_ground_truth = get_hidden(
            vanilla_model, test_iter, N=1000
        )
        vanilla_hidden_states = torch.tensor(vanilla_hidden_states).to(device)

        # Save the ground truth
        with open(load_path / "trained_ground_truth.npy", "wb") as f:
            np.save(f, trained_ground_truth)
        with open(load_path / "vanilla_ground_truth.npy", "wb") as f:
            np.save(f, vanilla_ground_truth)

        for rnn_layer in range(vanilla_hidden_states.shape[0]):
            save_path = load_path / str(rnn_layer + 1)
            save_path.mkdir(parents=True, exist_ok=True)

            pca = PCA()
            transformed = pca.fit_transform(vanilla_hidden_states[rnn_layer])
            with open(save_path / "pca_vanilla.pkl", "wb") as f:
                pickle.dump(pca, f)
            with open(save_path / "transformed_vanilla.npy", "wb") as f:
                np.save(f, transformed)

            pca = PCA(n_components=2)
            transformed = pca.fit_transform(vanilla_hidden_states[rnn_layer])
            with open(save_path / "pca_vanilla_n2.pkl", "wb") as f:
                pickle.dump(pca, f)
            with open(save_path / "transformed_vanilla_n2.npy", "wb") as f:
                np.save(f, transformed)

            pca = PCA()
            transformed = pca.fit_transform(trained_hidden_states[rnn_layer])
            with open(save_path / "pca_trained.pkl", "wb") as f:
                pickle.dump(pca, f)
            with open(save_path / "transformed_trained.npy", "wb") as f:
                np.save(f, transformed)

            pca = PCA(n_components=2)
            transformed = pca.fit_transform(trained_hidden_states[rnn_layer])
            with open(save_path / "pca_trained_n2.pkl", "wb") as f:
                pickle.dump(pca, f)
            with open(save_path / "transformed_trained_n2.npy", "wb") as f:
                np.save(f, transformed)

