import os
import sys
import torch
import torch.nn as nn

# Make sure Python can find the imported DeepLab repo code
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPLAB_DIR = os.path.join(CURRENT_DIR, "deeplab")

if DEEPLAB_DIR not in sys.path:
    sys.path.insert(0, DEEPLAB_DIR)

from network import modeling


class Model(nn.Module):
    """
    DeepLabV3+ segmentation model with a ResNet backbone.
    This replaces the original U-Net completely.
    """

    def __init__(
        self,
        in_channels=3,
        n_classes=19,
        backbone="resnet101",
        output_stride=16,
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError("This DeepLabV3+ setup expects 3-channel RGB input.")

        if backbone == "resnet50":
            self.model = modeling.deeplabv3plus_resnet50(
                num_classes=n_classes,
                output_stride=output_stride,
            )
        elif backbone == "resnet101":
            self.model = modeling.deeplabv3plus_resnet101(
                num_classes=n_classes,
                output_stride=output_stride,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.backbone_name = backbone
        self.output_stride = output_stride

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but got {x.shape[1]}"
            )
        return self.model(x)