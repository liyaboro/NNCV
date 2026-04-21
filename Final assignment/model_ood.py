import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure Python can find the imported DeepLab repo code
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPLAB_DIR = os.path.join(CURRENT_DIR, "deeplab")

if DEEPLAB_DIR not in sys.path:
    sys.path.insert(0, DEEPLAB_DIR)

from network import modeling

class ASPPFeatureVAE(nn.Module):
    """
    Lightweight convolutional VAE for ASPP features.
    Input:  [B, 256, H, W]
    Output: reconstructed feature map with same shape
    """

    def __init__(self, in_channels=256, latent_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)

        # Decoder seed
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)

        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, in_channels, kernel_size=3, padding=1),
        )

    def encode(self, x):
        h = self.enc(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z, output_size):
        h = self.fc_dec(z)
        h = h.view(z.shape[0], 64, 8, 8)
        h = self.dec(h)
        h = F.interpolate(h, size=output_size, mode="bilinear", align_corners=False)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, output_size=x.shape[-2:])
        return recon, mu, logvar


class Model(nn.Module):
    """
    DeepLabV3+ segmentation model with a VAE OOD head on ASPP features.
    Final forward returns:
        logits, include_decision
    """

    def __init__(
        self,
        in_channels=3,
        n_classes=19,
        backbone="resnet101",
        output_stride=16,
        latent_dim=64,
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
                pretrained_backbone=False,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.backbone_name = backbone
        self.output_stride = output_stride

        # VAE on ASPP-upsampled features
        self.vae = ASPPFeatureVAE(in_channels=256, latent_dim=latent_dim)

        # Stored threshold for OOD decision
        self.register_buffer("ood_threshold", torch.tensor(0.0, dtype=torch.float32))
    
    def forward_seg_with_aspp(self, x):
        """
        Runs the segmentation backbone/head manually so we can branch off
        the ASPP features before the final classifier.
        Returns:
            logits: [B, C, H, W]
            aspp_up: [B, 256, h, w]
        """
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but got {x.shape[1]}"
            )

        features = self.model.backbone(x)

        # DeepLabV3+ internals from classifier
        low_level_feature = self.model.classifier.project(features["low_level"])
        aspp_feature = self.model.classifier.aspp(features["out"])
        aspp_up = F.interpolate(
            aspp_feature,
            size=low_level_feature.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        fused = torch.cat([low_level_feature, aspp_up], dim=1)
        logits = self.model.classifier.classifier(fused)

        # Upsample back to input image size
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits, aspp_up
    
    def compute_ood_score(self, aspp_up):
        recon, mu, logvar = self.vae(aspp_up)

        #mean pooling
        score = ((aspp_up - recon) ** 2).mean(dim=(1, 2, 3))

        # #top-k pooling

        # # Per-pixel reconstruction error map:
        # # [B, C, H, W] -> [B, H, W]
        # error_map = ((aspp_up - recon) ** 2).mean(dim=1)

        # # Flatten spatial dimensions: [B, H, W] -> [B, H*W]
        # error_flat = error_map.flatten(start_dim=1)

        # # Take the top 5% highest-error locations
        # k = max(1, int(0.05 * error_flat.shape[1]))
        # topk_vals, _ = torch.topk(error_flat, k=k, dim=1)

        # # Image-level score = mean of top-k errors
        # score = topk_vals.mean(dim=1)

        return score, recon, mu, logvar
    
    def forward(self, x):
        """
        Final inference interface for submission:
            seg_logits, include_decision
        """
        logits, aspp_up = self.forward_seg_with_aspp(x)
        score, _, _, _ = self.compute_ood_score(aspp_up)
        include = score < self.ood_threshold
        return logits, include
    
    def forward_train_ood(self, x):
        """
        Helper for OOD/VAE training:
            returns logits, aspp features, recon, mu, logvar, score
        """
        logits, aspp_up = self.forward_seg_with_aspp(x)
        score, recon, mu, logvar = self.compute_ood_score(aspp_up)
        return logits, aspp_up, recon, mu, logvar, score
    
    def freeze_segmentation(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_segmentation(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def set_ood_threshold(self, value: float):
        self.ood_threshold.fill_(float(value))