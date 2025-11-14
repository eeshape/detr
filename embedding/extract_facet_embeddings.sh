#!/bin/bash
# FACET 데이터셋에서 임베딩 벡터 f를 추출하는 스크립트

# 사용 예시:
# 1. DETR 공식 pretrained 모델 사용하여 전체 임베딩 추출
# bash extract_facet_embeddings.sh --extract_type all

# 2. Decoder output (query embeddings)만 추출 (가장 유용)
# bash extract_facet_embeddings.sh --extract_type decoder

# 3. 특정 skin tone에 대해서만 추출
# bash extract_facet_embeddings.sh --skin skin1 --extract_type decoder

# 설정
COCO_PATH="/home/dohyeong/Desktop/COCO(split)/skin_group_1-2-3_split/images_test"
CHECKPOINT="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"  # DETR 공식 pretrained 모델
OUTPUT_DIR="/home/dohyeong/Desktop/COCO(split)/skin_group_1-2-3_split/images_test/embeddings_output"
BATCH_SIZE=32
EXTRACT_TYPE="all"  # backbone, encoder, decoder, all

# 인자 파싱
SKIN=""
LIGHT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --extract_type)
            EXTRACT_TYPE="$2"
            shift 2
            ;;
        --skin)
            SKIN="--skin $2"
            shift 2
            ;;
        --light)
            LIGHT="--light $2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 임베딩 추출 실행
cd "$(dirname "$0")"

python extract_embeddings.py \
    --batch_size $BATCH_SIZE \
    --no_aux_loss \
    --coco_path $COCO_PATH \
    --resume $CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --extract_type $EXTRACT_TYPE \
    $SKIN \
    $LIGHT \
    --num_workers 4

echo "임베딩 추출 완료: $OUTPUT_DIR"
