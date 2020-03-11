from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator
from torchtext.vocab import GloVe, FastText

TEXT = Field(sequential=True, lower=True, tokenize="spacy")
LABEL = Field(sequential=False, use_vocab=False)


def imdb(embedding=None):
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

    TEXT.build_vocab(train, vectors=vectors)

    # make iterator for splits
    train_iter, test_iter = BucketIterator.splits(
        (train, test), batch_size=32, device=0
    )

    return train_iter, test_iter
