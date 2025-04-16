import os
import glob
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F
import torch
from pycocotools.coco import COCO
from natsort import natsorted

class COCODataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.img_dir = os.path.join(root_dir, mode)
        self.anno_dir = os.path.join(root_dir, f'{mode}.json')
        self.transform = transform
        self.coco = COCO(self.anno_dir)
        self.ids = list(natsorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, self.coco.imgs[img_id]['file_name'])
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
    def __init__(self, root_dir, mode='test', transform=None):
        self.img_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.images = list(natsorted(os.listdir(self.img_dir)))
        self.images = sorted(self.images, key=lambda x: int(x.split(".")[0]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, idx
