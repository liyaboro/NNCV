wandb login

python3 -u train_ood.py \
    --data-dir ./data/cityscapes \
    --mode seg \
    --batch-size 32 \
    --epochs 40 \
    --lr 1e-4 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "deeplab-resnet101-ft-segv4" \
    --pretrained-ckpt ./checkpoints/deeplab_resnet101_os16_base.pth.tar

BEST_SEG_CKPT=$(ls -t checkpoints/deeplab-resnet101-ft-segv4/best_model-*.pt | head -n 1)
if [ -z "$BEST_SEG_CKPT" ]; then
    echo "No best segmentation checkpoint found."
    exit 1
fi

python3 -u train_ood.py \
    --data-dir ./data/cityscapes \
    --mode ood \
    --batch-size 32 \
    --epochs 20 \
    --lr 1e-4 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "deeplab-resnet101-ft-ood" \
    --pretrained-ckpt "$BEST_SEG_CKPT" 