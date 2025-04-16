import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from models.faster_rcnn import (
    CustomFasterRCNN,
    FasterRCNN_MobileNetV3_Large_320_FPN,
    FasterRCNN_MobileNetV3_Large_FPN,
    FasterRCNN_ResNest50_FPN,
    FasterRCNN_ResNet50_FPN,
    FasterRCNN_ResNext50_FPN,
)
from utils.dataloader import COCODataset, COCOTestDataset


MODEL_DICT = {
    "faster_rcnn": FasterRCNN_ResNet50_FPN,
    "faster_rcnn_resnest50_fpn": FasterRCNN_ResNest50_FPN,
    "faster_rcnn_resnext50_fpn": FasterRCNN_ResNext50_FPN,
    "faster_rcnn_mobilenetv3_large_fpn": FasterRCNN_MobileNetV3_Large_FPN,
    "faster_rcnn_mobilenetv3_large_320_fpn": FasterRCNN_MobileNetV3_Large_320_FPN,
    "faster_rcnn_custom": CustomFasterRCNN,
}


def collate_fn(batch):
    """Collate function for DataLoader."""
    return tuple(zip(*batch))


class Trainer:
    """Trainer class for object detection model."""

    def __init__(self, args):
        """Initialize trainer with arguments."""
        self.args = args
        os.makedirs(args.output_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=args.output_dir)

        self.transform_train = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
        ])
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.train_loader = DataLoader(
            COCODataset(args.data_dir, mode='train', transform=self.transform_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            collate_fn=collate_fn
        )
        self.valid_loader = DataLoader(
            COCODataset(args.data_dir, mode='valid', transform=self.transform_val),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            collate_fn=collate_fn
        )
        self.test_loader = DataLoader(
            COCOTestDataset(args.data_dir, mode='test', transform=self.transform_val),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            collate_fn=collate_fn
        )

        box_score_threshold = 0.5
        if args.eval_only:
            box_score_threshold = 0.7

        self.model = MODEL_DICT[args.model](
            box_score_threshold=box_score_threshold,
            num_classes=11,
        ).to(args.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params/1e6:.10f}M")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=1e-2
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=1e-6,
        )

        self.scaler = torch.amp.GradScaler()  # Mixed Precision Training
        self.best_accuracy = 0.0

    def train(self):
        """Train the model."""
        for epoch in range(self.args.epochs):
            train_losses = {
                "cls_loss": [],
                "box_loss": [],
                "obj_loss": [],
                "rpn_box_loss": [],
                "total_loss": [],
            }

            self.model.train()

            prefetcher = iter(self.train_loader)  # Data Prefetching
            for images, targets in tqdm(
                prefetcher,
                desc=f"Epoch {epoch+1}/{self.args.epochs}"
            ):
                images = list(image.to(self.args.device) for image in images)
                targets = [
                    {k: v.to(self.args.device) for k, v in t.items()}
                    for t in targets
                ]

                self.optimizer.zero_grad()
                with torch.autocast("cuda"):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                self.scaler.scale(losses).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0
                )  # Gradient Clipping
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_losses["cls_loss"].append(loss_dict["loss_classifier"].item())
                train_losses["box_loss"].append(loss_dict["loss_box_reg"].item())
                train_losses["obj_loss"].append(loss_dict["loss_objectness"].item())
                train_losses["rpn_box_loss"].append(
                    loss_dict["loss_rpn_box_reg"].item()
                )
                train_losses["total_loss"].append(losses.item())

            self.scheduler.step()

            # validate
            map_result = self.validate(self.valid_loader)

            # print log
            print(
                f"Epoch [{epoch+1}/{self.args.epochs}], "
                f"Loss: {np.mean(train_losses['total_loss']):.4f}, "
                f"mAP: {map_result['map']:.4f}"
            )

            if map_result["map"] > self.best_accuracy:
                ckpt_path = os.path.join(
                    self.args.output_dir,
                    "best_model.pth"
                )
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Saved best model to {ckpt_path}")
                self.best_accuracy = map_result["map"]

            # write to tensorboard
            self.writer.add_scalar(
                "Loss/cls_loss",
                np.mean(train_losses["cls_loss"]),
                epoch
            )
            self.writer.add_scalar(
                "Loss/box_loss",
                np.mean(train_losses["box_loss"]),
                epoch
            )
            self.writer.add_scalar(
                "Loss/obj_loss",
                np.mean(train_losses["obj_loss"]),
                epoch
            )
            self.writer.add_scalar(
                "Loss/rpn_box_loss",
                np.mean(train_losses["rpn_box_loss"]),
                epoch
            )
            self.writer.add_scalar(
                "Loss/total_loss",
                np.mean(train_losses["total_loss"]),
                epoch
            )
            self.writer.add_scalar("mAP", map_result["map"], epoch)

    def validate(self, loader):
        """Validate the model."""
        self.model.eval()
        with torch.no_grad():
            map_metric = MeanAveragePrecision()
            for images, targets in loader:
                images = list(image.to(self.args.device) for image in images)
                targets = [
                    {k: v.to(self.args.device) for k, v in t.items()}
                    for t in targets
                ]
                pred = self.model(images)
                map_metric.update(pred, targets)

        return map_metric.compute()

    def eval(self):
        """Evaluate the model on test set."""
        ckpt_path = os.path.join(self.args.output_dir, "best_model.pth")
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=self.args.device)
            )
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            print(
                f"Checkpoint {ckpt_path} not found. Exiting evaluation."
            )
            return

        self.model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, idx in tqdm(
                self.test_loader,
                desc="Testing",
                unit="batch",
            ):
                images = list(image.to(self.args.device) for image in images)
                pred = self.model(images)

                for i, p in zip(idx, pred):
                    digits = []
                    for j in range(len(p["boxes"])):
                        x_min, y_min, x_max, y_max = p["boxes"][j].cpu().tolist()

                        pred_dict = {
                            "image_id": i + 1,
                            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                            "score": p["scores"][j].item(),
                            "category_id": p["labels"][j].item(),
                        }
                        val_preds.append(pred_dict)

                        digits.append({"x_min": x_min, "val": p["labels"][j].item()})

                    if len(digits) > 0:
                        digits.sort(key=lambda d: d["x_min"])
                        pred_val = int("".join(str(d["val"] - 1) for d in digits))
                    else:
                        pred_val = -1

                    val_labels.append([i + 1, pred_val])

        # save predictions
        with open(os.path.join(self.args.output_dir, "pred.json"), "w") as f:
            json.dump(val_preds, f, indent=4)

        df = pd.DataFrame(val_labels, columns=["image_id", "pred_label"])
        df.to_csv(os.path.join(self.args.output_dir, "pred.csv"), index=False)

        # Create grid plot of first 10 images with predictions
        plt.figure(figsize=(20, 10))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            # Get original image tensor from dataset
            image_tensor = self.test_loader.dataset[i][0]
            # Convert to numpy for display
            image_np = image_tensor.permute(1, 2, 0).numpy()
            plt.imshow(image_np)
            
            # Get model predictions for this image
            with torch.no_grad():
                self.model.eval()
                # Move image to GPU (no need to add batch dimension)
                image_tensor = image_tensor.to(self.args.device)
                pred = self.model([image_tensor])
                boxes = pred[0]['boxes'].cpu().numpy()
                scores = pred[0]['scores'].cpu().numpy()
                labels = pred[0]['labels'].cpu().numpy()
                
                # Draw bounding boxes
                for box, score, label in zip(boxes, scores, labels):
                    if score > 0.5:  # Only draw boxes with confidence > 0.5
                        x1, y1, x2, y2 = box
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           fill=False, edgecolor='red', linewidth=2)
                        plt.gca().add_patch(rect)
                        plt.text(x1, y1, f'{label}:{score:.2f}', 
                               color='red', fontsize=8, 
                               bbox=dict(facecolor='white', alpha=0.5))
            
            plt.axis('off')
            plt.title(f'Image {i+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'first_10_images_with_boxes.png'))
        plt.close()
        print(f"Grid plot of first 10 images with bounding boxes saved to {os.path.join(self.args.output_dir, 'first_10_images_with_boxes.png')}")
