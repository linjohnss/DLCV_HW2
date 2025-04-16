import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Faster R-CNN model on COCO dataset'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to the COCO dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for data loading'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.005,
        help='Learning rate for training'
    )
    return parser.parse_args() 