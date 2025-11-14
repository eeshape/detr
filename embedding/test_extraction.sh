#!/bin/bash
# 테스트용: COCO 데이터셋에서 소량의 임베딩 추출하여 테스트

cd "$(dirname "$0")"

echo "=== DETR 임베딩 추출 테스트 ==="
echo ""
echo "DETR 공식 pretrained 모델로 소량의 이미지에서 임베딩을 추출합니다."
echo "전체 데이터셋 처리 전에 이 스크립트로 테스트하는 것을 권장합니다."
echo ""

# 설정
COCO_PATH="/home/dohyeong/Desktop/COCO"
CHECKPOINT="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
OUTPUT_DIR="./test_embeddings"
BATCH_SIZE=4
EXTRACT_TYPE="decoder"  # Query embeddings만 추출 (빠름)

echo "설정:"
echo "  데이터셋: $COCO_PATH"
echo "  체크포인트: $CHECKPOINT"
echo "  출력 디렉토리: $OUTPUT_DIR"
echo "  배치 크기: $BATCH_SIZE"
echo "  추출 타입: $EXTRACT_TYPE"
echo ""

# 실행
echo "임베딩 추출 시작..."
python extract_embeddings.py \
    --batch_size $BATCH_SIZE \
    --no_aux_loss \
    --resume $CHECKPOINT \
    --coco_path $COCO_PATH \
    --output_dir $OUTPUT_DIR \
    --extract_type $EXTRACT_TYPE \
    --num_workers 2

if [ $? -eq 0 ]; then
    echo ""
    echo "=== 테스트 성공! ==="
    echo ""
    echo "추출된 임베딩을 확인하려면:"
    echo "  python -c \"from analyze_embeddings import EmbeddingAnalyzer; a = EmbeddingAnalyzer('$OUTPUT_DIR'); print('Images:', len(a.metadata['image_ids'])); print('Keys:', a.metadata['embedding_keys'])\""
    echo ""
    echo "전체 데이터셋에서 추출하려면:"
    echo "  bash extract_facet_embeddings.sh"
else
    echo ""
    echo "=== 오류 발생 ==="
    echo "위의 오류 메시지를 확인하세요."
fi
