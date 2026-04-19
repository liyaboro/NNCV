wandb login

python3 -u train.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "deeplab-resnet101-ft-baseline-rc+dice+scheduler2" \
    --pretrained-ckpt ./checkpoints/deeplab_resnet101_os16_base.pth.tar