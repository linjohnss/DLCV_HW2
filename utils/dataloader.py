import os
from collections import defaultdict
from PIL import Image

import torch
from natsort import natsorted
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T


class COCODataset(Dataset):
    """COCO dataset for object detection."""

    def __init__(self, root_dir, mode='train', transform=None):
        """Initialize COCO dataset.

        Args:
            root_dir: Root directory of the dataset
            mode: Dataset mode (train/valid)
            transform: Transform to apply to images
        """
        self.img_dir = os.path.join(root_dir, mode)
        self.anno_dir = os.path.join(root_dir, f'{mode}.json')
        self.transform = transform
        self.coco = COCO(self.anno_dir)
        self.ids = list(natsorted(self.coco.imgs.keys()))

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.ids)

    def __getitem__(self, idx):
        """Get an image and its annotations.

        Args:
            idx: Index of the image

        Returns:
            image: Transformed image
            target: Dictionary containing boxes and labels
        """
        img_id = self.ids[idx]
        img_path = os.path.join(
            self.img_dir,
            self.coco.imgs[img_id]['file_name']
        )
        image = Image.open(img_path).convert("RGB")
        anns = self.coco.imgToAnns[img_id]
        boxes = []
        labels = []
        for ann in anns:
            xmin, ymin, width, height = ann['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transform:
            image = self.transform(image)
        return image, target


class COCOTestDataset(Dataset):
    """COCO test dataset for object detection."""

    def __init__(self, root_dir, mode='test', transform=None):
        """Initialize COCO test dataset.

        Args:
            root_dir: Root directory of the dataset
            mode: Dataset mode (test)
            transform: Transform to apply to images
        """
        self.img_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.images = list(natsorted(os.listdir(self.img_dir)))
        self.images = sorted(self.images, key=lambda x: int(x.split(".")[0]))

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get an image and its index.

        Args:
            idx: Index of the image

        Returns:
            image: Transformed image
            idx: Index of the image
        """
        image_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, idx
