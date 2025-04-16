import torch
import torchvision
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN, fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FastRCNNConvFCHead,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)

from torchvision.models.detection.backbone_utils import (
    BackboneWithFPN,
    _resnet_fpn_extractor,
)
from torchvision.models import resnet50, ResNet50_Weights, resnext50_32x4d
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import MultiScaleRoIAlign

def _adjust_anchors():
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)

class BackboneWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.layer1 = backbone[0:5]  # conv1 + layer1
        self.layer2 = backbone[5]    # layer2
        self.layer3 = backbone[6]    # layer3
        self.layer4 = backbone[7]    # layer4
        
    def forward(self, x):
        x = self.layer1(x)
        x1 = x
        x = self.layer2(x)
        x2 = x
        x = self.layer3(x)
        x3 = x
        x = self.layer4(x)
        x4 = x
        return {"0": x1, "1": x2, "2": x3, "3": x4}

class FasterRCNN_ResNet50_FPN(nn.Module):
    def __init__(self, box_score_threshold=0.5, num_classes=11):
        super(FasterRCNN_ResNet50_FPN, self).__init__()

        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            pretrained_backbone=True,
            box_score_thresh=box_score_threshold,
            min_size=600,
            max_size=800,
        )

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        return self.model(images, targets)

class FasterRCNN_MobileNetV3_Large_FPN(nn.Module):
    def __init__(self, box_score_threshold=0.5, num_classes=11):
        super(FasterRCNN_MobileNetV3_Large_FPN, self).__init__()

        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            pretrained_backbone=True,
            box_score_thresh=box_score_threshold,
            min_size=600,
            max_size=800,
        )

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        return self.model(images, targets)

class FasterRCNN_MobileNetV3_Large_320_FPN(nn.Module):
    def __init__(self, box_score_threshold=0.5, num_classes=11):
        super(FasterRCNN_MobileNetV3_Large_320_FPN, self).__init__()

        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
            pretrained_backbone=True,
            box_score_thresh=box_score_threshold,
        )

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        return self.model(images, targets)
    

class FasterRCNN_ResNest50_FPN(nn.Module):
    def __init__(self, box_score_threshold=0.5, num_classes=11):
        super(FasterRCNN_ResNest50_FPN, self).__init__()

        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            pretrained_backbone=True,
            box_score_thresh=box_score_threshold,
            min_size=600,
            max_size=800,
        )
        
        # Replace the backbone body with ResNeXt50
        torch.hub.list("zhanghang1989/ResNeSt", force_reload=True)
        new_backbone = torch.hub.load("zhanghang1989/ResNeSt", "resnest50", pretrained=True)
        new_backbone = nn.Sequential(*list(new_backbone.children())[:-2])        
        
        # Wrap the backbone
        wrapped_backbone = BackboneWrapper(new_backbone)
        self.model.backbone.body = wrapped_backbone
        
        # Update RPN anchor generator
        self.model.rpn.anchor_generator = _adjust_anchors()
        
        # Update RPN head
        in_channels = self.model.rpn.head.conv[0][0].in_channels
        self.model.rpn.head = RPNHead(
            in_channels=in_channels,
            num_anchors=self.model.rpn.anchor_generator.num_anchors_per_location()[0],
        )

        # Update box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.roi_heads.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3", "pool"],
            output_size=7,
            sampling_ratio=0
        )

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        return self.model(images, targets)



class FasterRCNN_ResNext50_FPN(nn.Module):
    def __init__(self, box_score_threshold=0.5, num_classes=11):
        super(FasterRCNN_ResNext50_FPN, self).__init__()
    
        # Initialize the base model
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            pretrained_backbone=True,
            box_score_thresh=box_score_threshold,
            min_size=600,
            max_size=800,
        )
        
        # Replace the backbone body with ResNeXt50
        new_backbone = resnext50_32x4d(weights='IMAGENET1K_V1')
        # Remove the last two layers (avgpool and fc)
        new_backbone = nn.Sequential(*list(new_backbone.children())[:-2])
        
        # Wrap the backbone
        wrapped_backbone = BackboneWrapper(new_backbone)
        self.model.backbone.body = wrapped_backbone
        
        # Update RPN anchor generator
        self.model.rpn.anchor_generator = _adjust_anchors()
        
        # Update RPN head
        in_channels = self.model.rpn.head.conv[0][0].in_channels
        self.model.rpn.head = RPNHead(
            in_channels=in_channels,
            num_anchors=self.model.rpn.anchor_generator.num_anchors_per_location()[0],
        )

        # Update box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.roi_heads.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3", "pool"],
            output_size=7,
            sampling_ratio=0
        )

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        return self.model(images, targets)

class CustomFasterRCNN(nn.Module):
    def __init__(self, box_score_threshold: float = 0.5, num_classes: int = 11):
        super().__init__()

        # 1. 建好原始模型（含 FPN）
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            pretrained_backbone=True,
            box_score_thresh=box_score_threshold,
            min_size=600,
            max_size=800,
        )

        # 2. 依需求調整 RPN
        self.model.rpn.anchor_generator = _adjust_anchors()
        in_channels = self.model.rpn.head.conv[0][0].in_channels
        self.model.rpn.head = RPNHead(
            in_channels=in_channels,
            num_anchors=self.model.rpn.anchor_generator.num_anchors_per_location()[0],
        )

        # 3. 換掉 box predictor（類別數 +1 含背景）
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.roi_heads.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3", "pool"],
            output_size=7,
            sampling_ratio=0
        )


    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        return self.model(images, targets)
