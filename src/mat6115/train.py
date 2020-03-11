from mat6115.dataset import imdb


def main(dataset, embedding):
    if dataset == "imdb":
        train_iter, test_iter = imdb(embedding)
    print(len(train_iter))
    print(len(test_iter))

