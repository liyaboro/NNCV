wandb login

python3 -u train.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 5 \
    --lr 1e-4 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "deeplabv3plus-resnet101-ft-baseline-test" \
    --pretrained-ckpt ./checkpoints/deeplab_resnet101_os16_base.pth.tar