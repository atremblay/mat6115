import argparse
from pathlib import Path
from mat6115 import train


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
        "-e",
        "--embedding",
        help="Initial embedding to use for words",
        default=None,
        choices=["glove", "fasttext", None],
    )

    parser_train.add_argument("-c", "--config", help="Config file path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tool == "train":
        train.main(
            dataset=args.dataset, embedding=args.embedding, config_file=args.config
        )
    print(args)
