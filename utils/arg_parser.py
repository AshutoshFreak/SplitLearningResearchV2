import argparse


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Split Learning Research Simulation entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--number-of-clients",
        type=int,
        default=20,
        metavar="C",
        help="Number of Clients",
    )
    parser.add_argument(
        "--server-side-tuning",
        # action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
        metavar="SST",
        help="State if server side tuning needs to be done",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        metavar="TB",
        help="Input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="Total number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        metavar="LR",
        help="Learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "--server-sigma",
        type=float,
        default=0,
        metavar="SS",
        help="Noise multiplier for central layers",
    )
    parser.add_argument(
        "-g",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="G",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(        # needs to be implemented
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="States dataset to be used",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2647,
        help="Random seed",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MNIST_CNN",
        help="Model you would like to train",
    )
    parser.add_argument(
        "--epoch-batch",
        type=str,
        default="5",
        help="Number of epochs after which next batch of clients should join",
    )
    args = parser.parse_args()
    return args