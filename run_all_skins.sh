#!/bin/bash

# Run DETR evaluation for all skin types (skin1 to skin10)

for i in {1..10}
do
    echo "=========================================="
    echo "Starting evaluation for skin${i}"
    echo "=========================================="
    
    python main.py \
        --batch_size 16 \
        --no_aux_loss \
        --eval \
        --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
        --coco_path /home/dohyeong/Desktop/COCO \
        --skin skin${i}
    
    echo ""
    echo "Completed evaluation for skin${i}"
    echo ""
done

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
