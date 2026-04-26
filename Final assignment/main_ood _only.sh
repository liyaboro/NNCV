wandb login

python3 -u train_ood.py \
    --data-dir ./data/cityscapes \
    --mode ood \
    --batch-size 32 \
    --epochs 40 \
    --lr 1e-4 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "deeplab-resnet101-ft-ood-v3-95" \
    --pretrained-ckpt ./checkpoints/deeplab-resnet101-ft-segv4/best_model-epoch=0036-val_loss=0.22219626978039742.pt