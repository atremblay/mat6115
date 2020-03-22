import torch
import argparse
from pathlib import Path
from mat6115 import train, analysis


def parse_args():
    """Parse arguments on command line"""

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help="Tools for semester project", dest="tool")
    parser_train = subparsers.add_parser("train", help="Train tool")

    parser_train.add_argument(
        "-s",
        "--save_path",
        help="Path where to save all the files (default .)",
        default=".",
        type=Path,
    )

    parser_train.add_argument(
        "-d",
        "--dataset",
        help="Supported dataset: imdb",
        default="imdb",
        choices=["imdb"],
    )
    parser_train.add_argument(
        "--rnn_type", help="Type of RNN layer", default="gru", choices=["gru", "rnn"],
    )

    parser_train.add_argument("--n_layers", default=1, type=int)

    parser_train.add_argument(
        "-e",
        "--embedding",
        help="Initial embedding to use for words",
        default="glove.6B.100d",
        choices=[
            "charngram.100d",
            "fasttext.en.300d",
            "fasttext.simple.300d",
            "glove.42B.300d",
            "glove.840B.300d",
            "glove.twitter.27B.25d",
            "glove.twitter.27B.50d",
            "glove.twitter.27B.100d",
            "glove.twitter.27B.200d",
            "glove.6B.50d",
            "glove.6B.100d",
            "glove.6B.200d",
            "glove.6B.300d",
        ],
    )

    parser_train.add_argument("--cuda", type=int, help="Cuda device to use")

    parser_artifacts = subparsers.add_parser(
        "artifacts", help="Create artifacts for further analysis."
    )

    parser_artifacts.add_argument(
        "-s",
        "--save_path",
        help="Path to the saved model. The artifacts will also be saved there.",
        required=True,
        type=Path,
    )

    parser_artifacts.add_argument(
        "-d",
        "--dataset",
        help="Supported dataset: imdb",
        default="imdb",
        choices=["imdb"],
    )

    parser_artifacts.add_argument(
        "-e",
        "--embedding",
        help="Initial embedding to use for words",
        default="glove.6B.100d",
        choices=[
            "charngram.100d",
            "fasttext.en.300d",
            "fasttext.simple.300d",
            "glove.42B.300d",
            "glove.840B.300d",
            "glove.twitter.27B.25d",
            "glove.twitter.27B.50d",
            "glove.twitter.27B.100d",
            "glove.twitter.27B.200d",
            "glove.6B.50d",
            "glove.6B.100d",
            "glove.6B.200d",
            "glove.6B.300d",
        ],
    )
    parser_artifacts.add_argument(
        "--rnn_layer",
        default=1,
        type=int,
        help="What RNN layer to use to create the artifacts",
    )

    parser_artifacts.add_argument("--fixed_point", action="store_true", default=False)

    parser_artifacts.add_argument("--cuda", type=int, help="Cuda device to use")
    parser_artifacts.add_argument(
        "--unique_fixed_point", action="store_true", default=False
    )
    parser_artifacts.add_argument("--pca", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.cuda is not None:
        device = torch.device("cuda", args.cuda)
    else:
        device = torch.device("cpu")

    if args.tool == "train":
        train.main(
            rnn_type=args.rnn_type,
            n_layers=args.n_layers,
            dataset=args.dataset,
            embedding=args.embedding,
            device=device,
        )
    elif args.tool == "artifacts":
        print("Running `artifacts` tool")
        analysis.main(
            save_path=args.save_path,
            dataset=args.dataset,
            embedding=args.embedding,
            rnn_layer=args.rnn_layer,
            device=device,
            fixed_point=args.fixed_point,
            unique_fixed_point=args.unique_fixed_point,
            pca=args.pca,
        )


if __name__ == "__main__":
    main()
