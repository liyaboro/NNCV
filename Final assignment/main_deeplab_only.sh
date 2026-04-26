wandb login

python3 -u train_ood.py \
    --data-dir ./data/cityscapes \
    --mode seg
    --batch-size 32 \
    --epochs 40 \
    --lr 1e-4 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "deeplab-resnet101-ft-segv4" \
    --pretrained-ckpt ./checkpoints/deeplab_resnet101_os16_base.pth.tar