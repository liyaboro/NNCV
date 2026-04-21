wandb login

python3 -u train_ood.py \
    --data-dir ./data/cityscapes \
    --mode ood \
    --batch-size 32 \
    --epochs 0 \
    --lr 1e-4 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "deeplab-resnet101-ft-ood2-70" \
    --ood-threshold-percentile 70.0 \
    --pretrained-ckpt ./checkpoints/deeplab-resnet101-ft-ood2-95/final_best_model-epoch=0028-val_loss=0.0070.pt