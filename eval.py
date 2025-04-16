import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Faster R-CNN model on COCO dataset'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model checkpoint'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to the COCO dataset directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for data loading'
    )
    return parser.parse_args() 