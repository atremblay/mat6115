from mat6115.model import Net
from mat6115.dataset import dataset_factory
from sklearn.decomposition import PCA
import pickle
import numpy as np


class BucketWrapper(object):

    """
    Tiny wrapper around the bucket iterator so we can use the
    `predict_generator` function from poutyne. That function expects
    an iterator that returns only inputs (i.e. no target)

    https://poutyne.org/model.html#poutyne.framework.Model.predict_generator

    """

    def __init__(self, bucket_iter, with_target=False):
        self._bucket_iter = bucket_iter

    def __len__(self):
        return len(self._bucket_iter)

    def __iter__(self):
        while True:
            for (x, y) in self._bucket_iter:
                if with_target:
                    yield x, y
                else:
                    yield x

    @property
    def batch_size(self):
        return self._bucket_iter.batch_size


def load_model(model_path):

    with open(model_path / "full_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    model.network.rnn.flatten_parameters()
    return model


def run_and_save_hidden(model, save_path, iterator, N=1000):
    batch_size = iterator.batch_size
    num_batch = N // batch_size

    _, _, preds = model.evaluate_generator(iterator, return_pred=True)
    _, outputs, output_lenghts = zip(*preds)
    flat_hidden_state = []
    num_hidden_states = 0

    for i, (_, output, output_lengths) in enumerate(preds):
        for o, length in zip(output, output_lengths):
            num_hidden_states += length
            flat_hidden_state.extend(o[:length])
        if N is not None and i >= num_batch:
            break

    flat_hidden_state = np.array(flat_hidden_state)
    assert flat_hidden_state.shape[0] == num_hidden_states
    np.save(save_path, flat_hidden_state)
    return flat_hidden_state


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
            try:
                num_hidden_states += length
                flat_hidden_state.append([o[:length] for o in output])
                ground_truth.extend([y_true[i][j]] * length)
                model_preds.extend([int(y_pred[j] > 0.0)] * length)
            except:
                __import__("pdb").set_trace()
                print(False)
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


def main(model_path, dataset, hidden_state_file):
    print("Loading model")
    model = load_model(model_path)
    print("Model loaded")
    print("Loading dataset")
    _, test_iter = dataset_factory(dataset)
    print("Dataset loaded")
    hidden_states = run_and_save_hidden(model, hidden_state_file, test_iter)


def analysis(hidden_states, pca_file, n_components=None):
    print("Running PCA")

    pca = PCA(n_components=n_components)
    reduced_hidden_state = pca.fit_transform(hidden_states)
    with open(pca_file, "wb") as pca_file:
        pickle.dump(pca, pca_file)

