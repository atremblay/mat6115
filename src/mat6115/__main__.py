import torch
import argparse
from pathlib import Path
from mat6115 import train, hidden


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

    parser_train.add_argument(
        "--analyze",
        help="Save the hidden states for the test set",
        default=False,
        action="store_true",
    )

    parser_train.add_argument("-c", "--config", help="Config file path", required=True)
    parser_train.add_argument("--cuda", type=int, help="Cuda device to use")

    parser_hidden = subparsers.add_parser(
        "hidden", help="Run the test set through the model and save the hidden states."
    )

    parser_hidden.add_argument(
        "-m", "--model_path", help="Path to the saved model.", required=True, type=Path,
    )

    parser_hidden.add_argument(
        "-d",
        "--dataset",
        help="Supported dataset: imdb",
        default="imdb",
        choices=["imdb"],
    )

    parser_hidden.add_argument(
        "-s",
        "--save_path",
        help="Path where to save all the files (default .)",
        default=".",
        type=Path,
    )

    parser_hidden.add_argument("-c", "--config", help="Config file path", required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.tool == "train":
        if args.cuda is not None:
            device = torch.device("cuda", args.cuda)
        else:
            device = torch.device("cpu")

        train.main(
            dataset=args.dataset,
            embedding=args.embedding,
            config_file=args.config,
            save_path=args.save_path,
            analyze=args.analyze,
            device=device,
        )
    elif args.tool == "hidden":
        print("Running `hidden` tool")
        hidden.main(
            model_path=args.model_path, dataset=args.dataset, save_path=args.save_path,
        )
    print(args)


if __name__ == "__main__":
    main()
