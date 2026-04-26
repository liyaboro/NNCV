# Semantic Segmentation with OOD Detection (Cityscapes)

This repository contains the implementation of a semantic segmentation model based on DeepLabV3+, extended with an Out-of-Distribution (OOD) detection head. The project focuses on robust urban scene understanding using the Cityscapes dataset.

---

## Repository Structure
```bash
Final assignment
├── deeplab/                # External DeepLabV3+ implementation
│   └── network/
│       └── modeling.py
│       └── ...
│   └── ...
├── .gitignore
├── Dockerfile
├── download_docker_and_data.sh 
├── jobscript_slurm.sh
├── main.double.sh
├── main_deeplab_only.sh
├── main_ood_only.sh
├── main.sh
├── model_ood_v1.py         # First version of OOD model (DeepLabV3+ + OOD head (no MSP))
├── model_ood.py            # Final model for OOD detection (DeepLabV3+ + OOD head with MSP)
├── model_unet.py           # U-Net model
├── model.py                # Final DeepLabV3+ model
├── predict_ood.py          # Inference script (segmentation + OOD)
├── predict.py              # Inference script (segmentation only)
├── README-x.md             # Given type specific README files (not very necessary unless you want details)
├── README.md               # This file
├── train_base.py           # Training script for baseline segmentation models
├── train_ood.py            # Training script for full model (DeepLabV3+ and VAE head)
├── train_v2.py             # Training script for second version segmentation model (addition of RandomCrop)
└── train.py                # Training script for last version segmentation model (RandomCrop + Dice + LR Scheduler)
```

---

## Installation (on HPC Cluster - SLURM)

All training experiments were performed on a high-performance computing (HPC) cluster using SLURM and containerized execution.

---

## Step 1: Clone Repository on the Cluster

Create a Personal Access Token (PAT) on GitHub:

- Go to: GitHub Settings → Developer Settings → Personal Access Tokens (PAT) → Tokens (classic)

Then log into the cluster and clone your repository:

```bash
git clone https://<PAT>@github.com/liyaboro/NNCV.git
cd NNCV
```
Replace `<PAT>` with your Personal Access Token.
Keep the repo up to date using 
```bash
git pull
```

>Note: Editing Code on the Cluster
>You can edit code directly on the cluster using VSCode:
>- Install Remote – SSH extension
>- Connect to the cluster via SSH
>- Open the repository folder
>- Edit, commit, and push changes

>This avoids syncing between local and remote environments.

## Step 2: Download Data and Container (One-Time Setup)
The training data and container are hosted externally.

Run the download script once using SLURM:

```bash
chmod +x download_docker_and_data.sh
sbatch download_docker_and_data.sh
```

After the job finishes, you should see:
- a `data/` directory
- a `container.sif` file

## Step 3: Configure Environment Variables and Dependencies

All experiments are executed inside a pre-built Singularity container (`container.sif`) provided with the assignment.

This container already includes all required dependencies (e.g. PyTorch, CUDA, etc.), so no additional installation via `requirements.txt` is necessary.

If you wish to run the code outside the container, you will need to manually install the required dependencies.

If running outside the container, the following dependencies are required:
```bash
wandb
torch
torchvision
numpy
pillow
tqdm
matplotlib
scikit-learn
pyyaml
```

To configure the environment variables, edit the .env file:
```bash
nano .env
```
Set required variables:
```bash
WANDB_API_KEY=<your_api_key>
WANDB_DIR=/home/<username>/wandb
```
These are used for experiment tracking.

## Step 4: Download pre-trained DeepLabV3+ checkpoint 

The pre-trained checkpoint that was used was downloaded on the following GitHub page: https://github.com/VainF/DeepLabV3Plus-Pytorch

Go to:
Results > Performance on Cityscapes (19 classes, 1024 x 2048) > DeepLabV3Plus-ResNet101 (click on the link).

Place the checkpoint file in the checkpoints/ folder (not inside an extra subfolder!)


## Step 5: Submit Training Job

Make the SLURM script executable and submit:
```bash
chmod +x jobscript_slurm.sh #only the first time
sbatch jobscript_slurm.sh
```
This script launches training inside the container using `main.sh` (entry point). Depending on which file you rename as `main.sh`, you will use different training files.
- `main_deeplab_only.sh`: trains only the segmentation model using `train_ood.py` (but if you remove --mode argument you can use `train.py`)
- `main_double.sh`: trains both segmentation model and then OOD on top of the just trained seg model.
- `main_deeplab_only.sh`: trains only the VAE head on top of a defained pre-trained seg model using `train_ood.py`
- `main_tresh.sh`: resets the threshold of the OOD detection using the final version

>Don't forget to adapt the `jobscript_slurm.sh` file to match the time you need. Training the segmentation model takes up to 2:30 hours for 50 epochs. Training the VAE head takes up to 00:30 minutes for 40 epochs. Threshold recomputation takes less than a minute. Always add some buffer time to ensure you are done training when it stops.

## Step 6: Evaluation

After training the model gets saved on the cluster into the `checkpoints/` folder, inside a subfolder names like the experiment-id.
OOD thresholds are computed automatically and saved into the .pt checkpoint file.

From there you can choose the model you want to conitnue with, and follow the instructions on the `README-Submission.md`, on how to submit the model to the server for evaluation on the test set.