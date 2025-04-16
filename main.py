import argparse
import random

import numpy as np
import torch

from utils.trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model", type=str, default="faster_rcnn")    

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Data directories
    parser.add_argument("--data_dir", type=str, default="./data")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="./output")

    # Evaluation options
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Only run evaluation, no training"
    )

    args = parser.parse_args()

    # Device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def main(args):
    torch.manual_seed(20250416)
    torch.cuda.manual_seed_all(20250416)
    np.random.seed(20250416)
    random.seed(20250416)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(args)

    if not args.eval_only:
        trainer.train()
    else:
        trainer.eval()


if __name__ == "__main__":
    args = get_args()
    main(args)
