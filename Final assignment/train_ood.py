"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
    RandomCrop
)
from torchvision.transforms import RandomCrop, CenterCrop
from torchvision.transforms.functional import crop
import torch.nn.functional as F
import numpy as np

from model_ood import Model


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image

class JointRandomCropCityscapes(torch.utils.data.Dataset):
    """
    Applies the same random crop to both image and target.
    Intended for training.
    """
    def __init__(self, base_dataset, crop_size=(512, 512)):
        self.base_dataset = base_dataset
        self.crop_size = crop_size

        self.img_transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.target_transform = Compose([
            ToImage(),
            ToDtype(torch.int64),
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]

        i, j, h, w = RandomCrop.get_params(image, output_size=self.crop_size)

        image = crop(image, i, j, h, w)
        target = crop(target, i, j, h, w)

        image = self.img_transform(image)
        target = self.target_transform(target)

        return image, target


class JointDeterministicValCityscapes(torch.utils.data.Dataset):
    """
    Applies a deterministic transform to both image and target.
    Intended for validation.

    If the image is large enough, use CenterCrop(crop_size).
    Otherwise, resize to crop_size.
    """
    def __init__(self, base_dataset, crop_size=(512, 512)):
        self.base_dataset = base_dataset
        self.crop_size = crop_size

        self.img_to_tensor = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.target_to_tensor = Compose([
            ToImage(),
            ToDtype(torch.int64),
        ])

        self.resize_img = Resize(crop_size, interpolation=InterpolationMode.BILINEAR)
        self.resize_target = Resize(crop_size, interpolation=InterpolationMode.NEAREST)

        self.center_crop = CenterCrop(crop_size)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]

        width, height = image.size
        crop_h, crop_w = self.crop_size

        if height >= crop_h and width >= crop_w:
            image = self.center_crop(image)
            target = self.center_crop(target)
        else:
            image = self.resize_img(image)
            target = self.resize_target(target)

        image = self.img_to_tensor(image)
        target = self.target_to_tensor(target)

        return image, target

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C, H, W]
        targets: [B, H, W] with values in {0, ..., C-1} or ignore_index
        """
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        valid_mask = (targets != self.ignore_index)  # [B, H, W]

        # Replace ignored labels temporarily so one_hot works
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        target_one_hot = F.one_hot(safe_targets, num_classes=self.num_classes)  # [B, H, W, C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Zero out ignored pixels in both prediction and target
        valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]
        probs = probs * valid_mask
        target_one_hot = target_one_hot * valid_mask

        # Compute Dice per class over batch + spatial dims
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        denominator = torch.sum(probs, dims) + torch.sum(target_one_hot, dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()

        return dice_loss
        
def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--pretrained-ckpt", type=str, default="", help="Path to pretrained DeepLabV3+ checkpoint")
    parser.add_argument("--mode", type=str, default="seg", choices=["seg", "ood"], help="Training mode: segmentation or OOD VAE head")
    parser.add_argument("--vae-latent-dim", type=int, default=64, help="Latent dimension for VAE head")
    parser.add_argument("--vae-beta", type=float, default=1e-4, help="Weight for KL term in VAE loss")
    parser.add_argument("--ood-threshold-percentile", type=float, default=95.0, help="Percentile of validation ID scores used as OOD threshold")
    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),
        tags=[args.mode],  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load raw Cityscapes datasets
    train_base_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
    )

    valid_base_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
    )

    # Final transform setup:
    # - training: joint random crop
    # - validation: joint deterministic crop/resize
    train_dataset = JointRandomCropCityscapes(
        train_base_dataset,
        crop_size=(512, 512),
    )

    valid_dataset = JointDeterministicValCityscapes(
        valid_base_dataset,
        crop_size=(512, 512),
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
        latent_dim=args.vae_latent_dim,
    ).to(device)

    if args.pretrained_ckpt:
        checkpoint = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=False)

        # Some checkpoints store the state dict directly,
        # others store it under "model_state"
        state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint

        missing_keys, unexpected_keys = model.model.load_state_dict(state_dict, strict=False)

        print("Loaded pretrained  segmentation checkpoint.", flush = True)
        print(f"Missing keys: {missing_keys}", flush = True)
        print(f"Unexpected keys: {unexpected_keys}", flush = True)

    # Define the loss functions
    #criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(num_classes=19, ignore_index=255)

    # Define the optimizer
    if args.mode == "seg":
        optimizer = AdamW(model.parameters(), lr=args.lr)
    else:
        model.freeze_segmentation()
        optimizer = AdamW(model.vae.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    best_epoch = -1

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}", flush = True)
        if args.mode == "seg":
            # -------------------------
            # Segmentation training
            # -------------------------
            # Training
            model.train()
            for i, (images, labels) in enumerate(train_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                optimizer.zero_grad()
                outputs = model.forward_seg_with_aspp(images)[0]
                #loss = criterion(outputs, labels)
                loss_ce = ce_loss(outputs, labels)
                loss_dice = dice_loss(outputs, labels)
                loss = loss_ce + 0.5*loss_dice
                loss.backward()
                optimizer.step()

                wandb.log({
                    f"{args.mode}/train_loss": loss.item(),
                    f"{args.mode}/train_ce_loss": loss_ce.item(),
                    f"{args.mode}/train_dice_loss": loss_dice.item(),
                    f"{args.mode}/learning_rate": optimizer.param_groups[0]['lr'],
                    f"{args.mode}/epoch": epoch + 1,
                }, step=epoch * len(train_dataloader) + i)
                
            # Validation
            model.eval()
            with torch.no_grad():
                valid_ce_losses = []
                valid_dice_losses = []
                valid_total_losses = []
                for i, (images, labels) in enumerate(valid_dataloader):

                    labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                    images, labels = images.to(device), labels.to(device)

                    labels = labels.long().squeeze(1)  # Remove channel dimension

                    outputs = model.forward_seg_with_aspp(images)[0]
                    #loss = criterion(outputs, labels)
                    loss_ce = ce_loss(outputs, labels)
                    loss_dice = dice_loss(outputs, labels)
                    loss = loss_ce + 0.5*loss_dice

                    valid_ce_losses.append(loss_ce.item())
                    valid_dice_losses.append(loss_dice.item())
                    valid_total_losses.append(loss.item())
                
                    if i == 0:
                        predictions = outputs.softmax(1).argmax(1)

                        predictions = predictions.unsqueeze(1)
                        labels_vis = labels.unsqueeze(1)

                        predictions = convert_train_id_to_color(predictions)
                        labels_vis = convert_train_id_to_color(labels_vis)

                        predictions_img = make_grid(predictions.cpu(), nrow=8)
                        labels_img = make_grid(labels_vis.cpu(), nrow=8)

                        predictions_img = predictions_img.permute(1, 2, 0).numpy()
                        labels_img = labels_img.permute(1, 2, 0).numpy()

                        wandb.log({
                            "predictions": [wandb.Image(predictions_img)],
                            "labels": [wandb.Image(labels_img)],
                        }, step=(epoch + 1) * len(train_dataloader) - 1)
                
                valid_ce_loss = sum(valid_ce_losses) / len(valid_ce_losses)
                valid_dice_loss = sum(valid_dice_losses) / len(valid_dice_losses)
                valid_loss = sum(valid_total_losses) / len(valid_total_losses)

                wandb.log({
                    f"{args.mode}/valid_loss": valid_loss,
                    f"{args.mode}/valid_ce_loss": valid_ce_loss,
                    f"{args.mode}/valid_dice_loss": valid_dice_loss,
                }, step=(epoch + 1) * len(train_dataloader) - 1)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    
                    if current_best_model_path:
                        os.remove(current_best_model_path)
                    current_best_model_path = os.path.join(
                        output_dir, 
                        f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                    )
                    torch.save(model.state_dict(), current_best_model_path)

        else:
        # -------------------------
        # OOD / VAE-head training
        # -------------------------
            model.train()
            for i, (images, _) in enumerate(train_dataloader):
                images = images.to(device)

                optimizer.zero_grad()

                with torch.no_grad():
                    _, aspp_up = model.forward_seg_with_aspp(images)

                recon, mu, logvar = model.vae(aspp_up)

                rec_loss = F.mse_loss(recon, aspp_up)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = rec_loss + args.vae_beta * kl_loss

                loss.backward()
                optimizer.step()

                wandb.log({
                    f"{args.mode}/train_loss": loss.item(),
                    f"{args.mode}/train_rec_loss": rec_loss.item(),
                    f"{args.mode}/train_kl_loss": kl_loss.item(),
                    f"{args.mode}/learning_rate": optimizer.param_groups[0]['lr'],
                    f"{args.mode}/epoch": epoch + 1,
                }, step=epoch * len(train_dataloader) + i)

            # Validation
            model.eval()
            with torch.no_grad():
                valid_total_losses = []
                valid_rec_losses = []
                valid_kl_losses = []

                for i, (images, _) in enumerate(valid_dataloader):
                    images = images.to(device)

                    _, aspp_up = model.forward_seg_with_aspp(images)
                    recon, mu, logvar = model.vae(aspp_up)

                    rec_loss = F.mse_loss(recon, aspp_up)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = rec_loss + args.vae_beta * kl_loss

                    valid_total_losses.append(loss.item())
                    valid_rec_losses.append(rec_loss.item())
                    valid_kl_losses.append(kl_loss.item())

                valid_loss = sum(valid_total_losses) / len(valid_total_losses)
                valid_rec_loss = sum(valid_rec_losses) / len(valid_rec_losses)
                valid_kl_loss = sum(valid_kl_losses) / len(valid_kl_losses)

                wandb.log({
                    f"{args.mode}/valid_loss": valid_loss,
                    f"{args.mode}/valid_rec_loss": valid_rec_loss,
                    f"{args.mode}/valid_kl_loss": valid_kl_loss,
                }, step=(epoch + 1) * len(train_dataloader) - 1)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    if current_best_model_path:
                        os.remove(current_best_model_path)
                    current_best_model_path = os.path.join(
                        output_dir,
                        f"best_ood_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                    )
                    torch.save(model.state_dict(), current_best_model_path)

        scheduler.step()
            
    print(f"Training complete! Best validation loss: {best_valid_loss:04}", flush=True)

    if args.mode == "ood":
        # Load the best OOD model back into memory first
        print(f"Reloading best OOD model from: {current_best_model_path}", flush=True)
        best_state_dict = torch.load(current_best_model_path, map_location=device)
        model.load_state_dict(best_state_dict, strict=True)

        # Compute threshold ONLY for this best model
        print("Computing OOD threshold from validation ID scores for BEST model...", flush=True)
        model.eval()
        val_scores = []

        with torch.no_grad():
            for images, _ in valid_dataloader:
                images = images.to(device)

                _, aspp_up = model.forward_seg_with_aspp(images)
                recon, _, _ = model.vae(aspp_up)
                scores = ((aspp_up - recon) ** 2).mean(dim=(1, 2, 3))

                val_scores.extend(scores.cpu().numpy().tolist())

        threshold = float(np.percentile(val_scores, args.ood_threshold_percentile))
        model.set_ood_threshold(threshold)

        wandb.log({
            "ood/threshold": threshold,
        })

        print(f"Computed OOD threshold for BEST model: {threshold:.8f}", flush=True)


        # Save the model
        submission_model_path = os.path.join(output_dir, f"final_best_model-epoch={best_epoch:04}-val_loss={best_valid_loss:.4f}.pt")
        torch.save(model.state_dict(), submission_model_path)
        print(f"Saved submission-ready model to:{submission_model_path}", flush=True)
    else:
        # For segmentation-only runs, keep saving the final model if you want
        final_model_path = os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
        )
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final segmentation model to: {final_model_path}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
