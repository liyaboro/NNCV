wandb login

python3 -u train_ood.py \
    --data-dir ./data/cityscapes \
    --mode ood \
    --batch-size 32 \
    --epochs 0 \
    --lr 1e-4 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "deeplab-resnet101-ft-ood-v2-95" \
    --ood-threshold-percentile 95.0 \
    --pretrained-ckpt ./checkpoints/deeplab-resnet101-ft-ood-v2-90/best_ood_model-epoch=0027-val_loss=0.006985788146266714.pt