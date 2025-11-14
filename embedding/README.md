# FACET 데이터셋에서 DETR 임베딩 벡터 f 추출하기

이 문서는 FACET 데이터셋을 DETR 모델에 통과시켜 임베딩 벡터 **f**를 추출하는 방법을 설명합니다.

> **빠른 시작**: [QUICKSTART.md](QUICKSTART.md)를 먼저 읽어보세요!

## 목차

1. [개요](#개요)
2. [파일 구조](#파일-구조)
3. [사용 방법](#사용-방법)
4. [활용 예시](#활용-예시)
5. [문제 해결](#문제-해결)

## 개요

DETR (DEtection TRansformer) 모델에서 추출할 수 있는 주요 임베딩 벡터들:

1. **Backbone Features** (`backbone_features`)
   - ResNet 등 backbone을 통과한 feature map
   - Shape: `[batch_size, channels, height, width]`
   - 낮은 수준의 시각적 특징

2. **Encoder Output** (`encoder_output`)
   - Transformer encoder를 통과한 feature
   - Shape: `[height*width, batch_size, hidden_dim]`
   - 전역적인 context를 포함한 feature

3. **Query Embeddings** (`query_embeddings`) **← 가장 중요!**
   - Transformer decoder의 최종 출력 (object queries)
   - Shape: `[batch_size, num_queries, hidden_dim]`
   - 각 object에 대한 고수준 semantic 임베딩
   - **대부분의 downstream task에 가장 유용**

4. **Decoder Output** (`decoder_output`)
   - 모든 decoder layer의 hidden states
   - Shape: `[num_layers, batch_size, num_queries, hidden_dim]`

## 파일 구조

```
detr/
├── embedding/                          # 임베딩 추출 관련 모든 파일
│   ├── extract_embeddings.py           # 임베딩 추출 메인 스크립트
│   ├── extract_facet_embeddings.sh     # 실행용 bash 스크립트
│   ├── analyze_embeddings.py           # 추출된 임베딩 분석 도구
│   ├── demo_extract_single_image.py    # 단일 이미지 임베딩 추출 데모
│   └── README.md                       # 이 파일
├── models/
├── datasets/
└── ...
```

## 사용 방법

### 1. 임베딩 추출

#### 방법 A: Python 직접 실행 (main.py 스타일)

```bash
cd embedding

# DETR 공식 pretrained 모델 사용
python extract_embeddings.py \
    --batch_size 16 \
    --no_aux_loss \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /home/dohyeong/Desktop/FACET\(원본\) \
    --output_dir ./embeddings_output \
    --extract_type all \
    --num_workers 4

# 또는 로컬 체크포인트 사용
python extract_embeddings.py \
    --batch_size 16 \
    --no_aux_loss \
    --resume /path/to/your/checkpoint.pth \
    --coco_path /home/dohyeong/Desktop/COCO \
    --output_dir ./embeddings_output \
    --extract_type decoder
```

#### 방법 B: Bash 스크립트 사용

```bash
cd embedding

# 전체 임베딩 추출
bash extract_facet_embeddings.sh --extract_type all

# Query embeddings만 추출 (권장)
bash extract_facet_embeddings.sh --extract_type decoder

# 특정 skin tone 그룹만
bash extract_facet_embeddings.sh --skin skin1 --extract_type decoder

# Lighting 조건 필터링
bash extract_facet_embeddings.sh --light well --extract_type decoder
```

#### 주요 인자 설명

- `--coco_path`: FACET 데이터셋 경로 (필수)
- `--resume`: DETR 모델 체크포인트 경로 (필수)
  - URL 사용 가능: `https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth`
  - 로컬 파일: `/path/to/checkpoint.pth`
- `--output_dir`: 임베딩을 저장할 디렉토리
- `--batch_size`: 배치 크기 (기본값: 16)
- `--no_aux_loss`: auxiliary loss 비활성화 (evaluation시 필수)
- `--extract_type`: 추출할 임베딩 타입
  - `all`: 모든 임베딩 (backbone, encoder, decoder)
  - `decoder`: decoder output만 (query embeddings) - 권장
  - `encoder`: encoder output만
  - `backbone`: backbone features만
- `--skin`: 특정 skin tone 그룹 필터링 (skin1, skin2, ..., skin10)
- `--light`: lighting 조건 필터링 (well, dimly)
- `--num_workers`: 데이터 로딩 워커 수

### 2. 저장된 임베딩 구조

임베딩은 다음과 같은 구조로 저장됩니다:

```
embeddings_output/
├── metadata.json                    # 메타데이터
├── embedding_000000000001.pt        # 이미지 ID 1의 임베딩
├── embedding_000000000002.pt        # 이미지 ID 2의 임베딩
└── ...
```

각 `.pt` 파일은 다음을 포함하는 딕셔너리입니다:

```python
{
    'backbone_features': Tensor,     # [C, H, W]
    'encoder_output': Tensor,        # [H*W, C]
    'decoder_output': Tensor,        # [num_layers, num_queries, C]
    'query_embeddings': Tensor,      # [num_queries, C] ← 가장 중요!
    'pred_logits': Tensor,           # [num_queries, num_classes+1]
    'pred_boxes': Tensor,            # [num_queries, 4]
}
```

### 3. 임베딩 로드 및 사용

```python
import torch
from analyze_embeddings import EmbeddingAnalyzer

# 분석기 초기화
analyzer = EmbeddingAnalyzer('./embeddings_output')

# 메타데이터 확인
print(analyzer.metadata)

# 특정 이미지의 임베딩 로드
image_id = 1
embedding = analyzer.load_embedding(image_id)
query_emb = embedding['query_embeddings']  # [100, 256]

# 모든 query embeddings를 행렬로 로드
matrix, image_ids, query_indices = analyzer.get_query_embeddings_as_matrix()
# matrix shape: [num_images * num_queries, embedding_dim]
print(f"Total query embeddings: {matrix.shape}")
```

### 4. 임베딩 시각화 및 분석

```python
from analyze_embeddings import EmbeddingAnalyzer

analyzer = EmbeddingAnalyzer('./embeddings_output')

# 통계 정보
stats = analyzer.get_embedding_statistics('query_embeddings')
print(f"Shape: {stats['shape']}")
print(f"Mean norm: {stats['norm_mean']:.4f}")

# PCA 시각화
analyzer.visualize_embeddings_pca(
    'query_embeddings',
    n_components=2,
    save_path='pca_viz.png'
)

# t-SNE 시각화
analyzer.visualize_embeddings_tsne(
    'query_embeddings',
    n_components=2,
    save_path='tsne_viz.png'
)
```

## 활용 예시

### 1. Downstream Task용 Feature 추출

```python
import torch

# 모든 query embeddings를 numpy array로
analyzer = EmbeddingAnalyzer('./facet_embeddings')
embeddings, image_ids = analyzer.load_all_embeddings('query_embeddings')

# 각 이미지를 단일 벡터로 표현 (평균 pooling)
image_features = []
for emb in embeddings:
    # emb shape: [num_queries, embedding_dim]
    # 평균을 취하여 단일 벡터로
    image_vec = emb.mean(dim=0)  # [embedding_dim]
    image_features.append(image_vec.numpy())

import numpy as np
X = np.stack(image_features, axis=0)  # [num_images, embedding_dim]

# 이제 X를 사용하여 classification, clustering 등 수행
```

### 2. Object-level Analysis

```python
# 각 detected object의 임베딩 분석
embedding = analyzer.load_embedding(image_id)

query_emb = embedding['query_embeddings']  # [100, 256]
pred_logits = embedding['pred_logits']      # [100, 2] (person class만)
pred_boxes = embedding['pred_boxes']        # [100, 4]

# confidence가 높은 detection만 선택
confidence = pred_logits.softmax(-1)[:, 0]  # person class
high_conf_mask = confidence > 0.5

selected_embeddings = query_emb[high_conf_mask]
selected_boxes = pred_boxes[high_conf_mask]

# 각 detected person의 임베딩 사용
for i, (emb, box) in enumerate(zip(selected_embeddings, selected_boxes)):
    print(f"Person {i}: embedding dim={emb.shape}, box={box}")
```

### 3. Skin Tone별 임베딩 비교

```bash
# 각 skin tone 그룹별로 임베딩 추출
cd embedding

for skin in skin1 skin2 skin3 skin4 skin5 skin6 skin7 skin8 skin9 skin10; do
    python extract_embeddings.py \
        --batch_size 16 \
        --no_aux_loss \
        --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
        --coco_path /home/dohyeong/Desktop/FACET\(원본\) \
        --output_dir ./embeddings_${skin} \
        --extract_type decoder \
        --skin ${skin}
done
```

```python
# Python에서 비교
analyzers = {}
for i in range(1, 11):
    skin = f'skin{i}'
    analyzers[skin] = EmbeddingAnalyzer(f'./embeddings_{skin}')

# 각 그룹의 평균 임베딩 계산
mean_embeddings = {}
for skin, analyzer in analyzers.items():
    matrix, _, _ = analyzer.get_query_embeddings_as_matrix()
    mean_embeddings[skin] = matrix.mean(axis=0)

# 그룹 간 유사도 계산
from scipy.spatial.distance import cosine

for skin1 in mean_embeddings:
    for skin2 in mean_embeddings:
        if skin1 < skin2:
            sim = 1 - cosine(mean_embeddings[skin1], mean_embeddings[skin2])
            print(f"{skin1} vs {skin2}: similarity = {sim:.4f}")
```

## 주의사항

1. **메모리 사용량**: `--extract_type all`은 많은 메모리를 사용합니다. GPU 메모리가 부족하면 batch_size를 줄이거나 decoder만 추출하세요.

2. **저장 공간**: 전체 데이터셋의 임베딩은 수 GB를 차지할 수 있습니다.

3. **체크포인트**: 학습된 DETR 모델 체크포인트가 필요합니다. `--resume` 인자로 지정하세요.

4. **Query Embeddings 권장**: 대부분의 경우 `query_embeddings`가 가장 유용합니다. 이는 각 object에 대한 고수준 semantic 정보를 포함합니다.

## 문제 해결

### CUDA out of memory
```bash
# batch_size를 줄이기
python extract_embeddings.py --batch_size 8 ...
```

### 특정 임베딩만 필요한 경우
```bash
# decoder output만 추출 (메모리 절약)
python extract_embeddings.py --extract_type decoder ...
```

### 체크포인트 문제
```bash
# DETR 공식 pretrained 모델 사용 (권장)
python extract_embeddings.py \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --batch_size 16 \
    --no_aux_loss \
    ...
```

### 임베딩 차원 확인
```python
analyzer = EmbeddingAnalyzer('./facet_embeddings')
emb = analyzer.load_embedding(1)
for key, value in emb.items():
    print(f"{key}: {value.shape}")
```

## 추가 정보

- Query embeddings의 기본 차원: 256 (hidden_dim)
- 기본 query 개수: 100 (num_queries)
- Backbone feature의 기본 채널 수: 2048 (ResNet-50 사용 시)

임베딩 벡터 f를 활용하여 다양한 downstream task (classification, retrieval, fairness analysis 등)를 수행할 수 있습니다!
