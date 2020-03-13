import torch
from torchtext.data import BucketIterator, Field
from torchtext.datasets import IMDB
from torchtext.vocab import FastText, GloVe

TEXT = Field(sequential=True, lower=True, tokenize="spacy", batch_first=True)
LABEL = Field(sequential=False, use_vocab=True, batch_first=True)


class BucketWrapper(object):

    """
    Something weird is happening with the class BucketIterator, it does not
    seem to yield the proper tuple. So this class is to circumvent this.

    At the same time it converts the labels to 0 and 1. The IMDB class
    provided by torchtext returns the labels 'pos' and 'neg' instead
    of 1 and 0. Leaving the transformation to the label Field
    (with use_vocab = True)

    """

    def __init__(self, bucket_iter):
        self._bucket_iter = bucket_iter

    def __len__(self):
        return len(self._bucket_iter)

    def __iter__(self):
        while True:
            for (x, y), _ in self._bucket_iter:
                yield x, (y - 1.0).unsqueeze(-1)

    @property
    def batch_size(self):
        return self._bucket_iter.batch_size


def imdb(embedding=None):
    global LABEL
    LABEL = Field(sequential=False, use_vocab=True)

    # make splits for data
    train, test = IMDB.splits(TEXT, LABEL)

    # build the vocabulary

    if embedding == "glove":
        vectors = GloVe(name="6B", dim=300)
    elif embedding == "fasttext":
        vectors = FastText(language="en")
    elif embedding is None:
        vectors = None
    else:
        raise Exception(f"Embedding {embedding} not supported")

    TEXT.build_vocab(train, vectors=vectors, specials=["<pad>", "<null>"])

    # Need to build the vocab for the labels because they are `pos` and `neg`
    # This will convert them to numerical values
    LABEL.build_vocab(train)

    # make iterator for splits
    train_iter, test_iter = BucketIterator.splits(
        (train, test), batch_size=32, device=torch.device("cpu")
    )

    return BucketWrapper(train_iter), BucketWrapper(test_iter)
