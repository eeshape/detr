# DETR 임베딩 추출 빠른 시작 가이드

FACET 데이터셋에서 DETR 임베딩 벡터 **f**를 추출하는 방법입니다.

## 1. 간단한 실행 예제

### FACET 데이터셋에서 임베딩 추출

```bash
cd /home/dohyeong/Desktop/detr/embedding

# DETR 공식 pretrained 모델 사용
python extract_embeddings.py \
    --batch_size 16 \
    --no_aux_loss \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /home/dohyeong/Desktop/COCO \
    --output_dir ./embeddings_output \
    --extract_type all
```

### COCO 데이터셋에서 임베딩 추출

```bash
cd /home/dohyeong/Desktop/detr/embedding

python extract_embeddings.py \
    --batch_size 16 \
    --no_aux_loss \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /home/dohyeong/Desktop/COCO \
    --output_dir ./embeddings_coco \
    --extract_type decoder
```

## 2. 다양한 추출 옵션

### Query Embeddings만 추출 (권장 - 가장 유용)
```bash
python extract_embeddings.py \
    --batch_size 16 \
    --no_aux_loss \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /home/dohyeong/Desktop/FACET\(원본\) \
    --output_dir ./embeddings_query_only \
    --extract_type decoder
```

### 특정 Skin Tone 그룹만
```bash
python extract_embeddings.py \
    --batch_size 16 \
    --no_aux_loss \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /home/dohyeong/Desktop/FACET\(원본\) \
    --output_dir ./embeddings_skin1 \
    --extract_type decoder \
    --skin skin1
```

### Lighting 조건별
```bash
# Well-lit만
python extract_embeddings.py \
    --batch_size 16 \
    --no_aux_loss \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /home/dohyeong/Desktop/FACET\(원본\) \
    --output_dir ./embeddings_well_lit \
    --extract_type decoder \
    --light well

# Dimly-lit만
python extract_embeddings.py \
    --batch_size 16 \
    --no_aux_loss \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /home/dohyeong/Desktop/FACET\(원본\) \
    --output_dir ./embeddings_dimly_lit \
    --extract_type decoder \
    --light dimly
```

## 3. Bash 스크립트 사용

```bash
cd /home/dohyeong/Desktop/detr/embedding

# 기본 실행
bash extract_facet_embeddings.sh

# 옵션 지정
bash extract_facet_embeddings.sh --extract_type decoder

# Skin tone 지정
bash extract_facet_embeddings.sh --skin skin1 --extract_type decoder
```

## 4. 추출된 임베딩 사용하기

```python
import torch
from analyze_embeddings import EmbeddingAnalyzer

# 임베딩 로드
analyzer = EmbeddingAnalyzer('./embeddings_output')

# 메타데이터 확인
print(analyzer.metadata)

# 특정 이미지 임베딩 로드
image_id = analyzer.metadata['image_ids'][0]
embedding = analyzer.load_embedding(image_id)

# Query embeddings 추출 (가장 중요!)
query_emb = embedding['query_embeddings']  # Shape: [100, 256]
print(f"Query embeddings shape: {query_emb.shape}")

# 모든 이미지의 query embeddings를 행렬로
matrix, image_ids, query_indices = analyzer.get_query_embeddings_as_matrix()
print(f"Total embeddings: {matrix.shape}")  # [num_images * 100, 256]
```

## 5. 추출 타입별 설명

- `--extract_type all`: 모든 임베딩 (backbone, encoder, decoder)
- `--extract_type decoder`: **Query embeddings (가장 유용!)** - 권장
- `--extract_type encoder`: Encoder output만
- `--extract_type backbone`: Backbone features만

## 6. 주의사항

### GPU 메모리 부족 시
```bash
# batch_size를 줄이세요
python extract_embeddings.py --batch_size 8 ...
```

### 필수 인자
- `--coco_path`: 데이터셋 경로 (필수)
- `--resume`: 체크포인트 경로 (필수)
- `--no_aux_loss`: evaluation 시 필수

### 체크포인트 옵션
- DETR 공식: `https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth`
- 로컬 파일: `/path/to/your/checkpoint.pth`

## 7. 결과물

```
embeddings_output/
├── metadata.json              # 메타데이터
├── embedding_000000000001.pt  # 각 이미지의 임베딩
├── embedding_000000000002.pt
└── ...
```

각 `.pt` 파일에는 다음이 포함됩니다:
- `query_embeddings`: [100, 256] - **가장 중요!**
- `pred_logits`: [100, num_classes]
- `pred_boxes`: [100, 4]
- 기타 (`backbone_features`, `encoder_output` 등 - extract_type에 따라)

## 8. 더 자세한 정보

자세한 사용법과 예제는 `README.md`를 참고하세요.
