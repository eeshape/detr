# DETR 임베딩 추출 Output 구조 상세 설명

## 📁 Output 폴더 구조

### 기본 구조
```
embeddings_output/                    # --output_dir로 지정한 폴더
├── metadata.json                     # 메타데이터 파일
├── embedding_000000000001.pt         # 이미지 ID 1의 임베딩
├── embedding_000000000002.pt         # 이미지 ID 2의 임베딩
├── embedding_000000000003.pt         # 이미지 ID 3의 임베딩
└── ...                               # 각 이미지마다 하나씩
```

### 실제 예시 (100개 이미지 추출 시)
```
embeddings_output/
├── metadata.json                     # 1개
├── embedding_000000000001.pt         # 약 400KB
├── embedding_000000000002.pt         # 약 400KB
├── embedding_000000000003.pt         # 약 400KB
├── ...
└── embedding_000000000100.pt         # 약 400KB

총 파일: 101개 (metadata.json + 100개 .pt 파일)
총 용량: 약 40MB (100 이미지 * 400KB)
```

---

## 📄 1. metadata.json

JSON 형식의 메타정보 파일입니다.

### 내용
```json
{
  "image_ids": [1, 2, 3, 4, 5, ..., 100],
  "num_images": 100,
  "extract_type": "decoder",
  "embedding_keys": [
    "decoder_output",
    "query_embeddings",
    "pred_logits",
    "pred_boxes"
  ]
}
```

### 필드 설명
- **image_ids**: 추출된 모든 이미지의 ID 목록
- **num_images**: 총 이미지 개수
- **extract_type**: 사용한 추출 타입 (all, decoder, encoder, backbone)
- **embedding_keys**: 각 .pt 파일에 포함된 키 목록

---

## 💾 2. embedding_XXXXXXXXXXXX.pt 파일들

각 이미지마다 하나의 PyTorch 파일 (.pt)이 생성됩니다.

### 파일명 규칙
```
embedding_{image_id:012d}.pt
```
- 이미지 ID를 12자리로 zero-padding
- 예: 이미지 ID 1 → `embedding_000000000001.pt`
- 예: 이미지 ID 12345 → `embedding_000000012345.pt`

### 파일 형식
PyTorch 텐서를 포함하는 딕셔너리 (pickle 형식)

---

## 📊 3. 각 .pt 파일의 내용

### extract_type='decoder' 사용 시 (권장)

```python
{
    'decoder_output': torch.Tensor,      # Shape: [6, 100, 256]
    'query_embeddings': torch.Tensor,    # Shape: [100, 256] ← 가장 중요!
    'encoder_output': torch.Tensor,      # Shape: [H*W, 256] (예: [1600, 256])
    'pred_logits': torch.Tensor,         # Shape: [100, 92]
    'pred_boxes': torch.Tensor,          # Shape: [100, 4]
}
```

**Shape 설명:**
- `[6, 100, 256]`: [decoder layers, num_queries, hidden_dim]
- `[100, 256]`: [num_queries, hidden_dim]
- `[1600, 256]`: [height*width, hidden_dim] (feature map 크기에 따라 다름)
- `[100, 92]`: [num_queries, num_classes+1] (COCO는 91 classes)
- `[100, 4]`: [num_queries, box_coords] (cx, cy, w, h)

### extract_type='all' 사용 시

```python
{
    'backbone_features': torch.Tensor,   # Shape: [2048, H, W] (예: [2048, 25, 34])
    'encoder_output': torch.Tensor,      # Shape: [H*W, 256] (예: [850, 256])
    'decoder_output': torch.Tensor,      # Shape: [6, 100, 256]
    'query_embeddings': torch.Tensor,    # Shape: [100, 256] ← 가장 중요!
    'pred_logits': torch.Tensor,         # Shape: [100, 92]
    'pred_boxes': torch.Tensor,          # Shape: [100, 4]
}
```

### extract_type='encoder' 사용 시

```python
{
    'encoder_output': torch.Tensor,      # Shape: [H*W, 256]
    'pred_logits': torch.Tensor,         # Shape: [100, 92]
    'pred_boxes': torch.Tensor,          # Shape: [100, 4]
}
```

### extract_type='backbone' 사용 시

```python
{
    'backbone_features': torch.Tensor,   # Shape: [2048, H, W]
    'pred_logits': torch.Tensor,         # Shape: [100, 92]
    'pred_boxes': torch.Tensor,          # Shape: [100, 4]
}
```

---

## 🔍 4. 각 텐서의 의미

### 1) query_embeddings (가장 중요!) 🌟
```
Shape: [100, 256]
Type: torch.FloatTensor
```
- **의미**: 각 object query의 최종 임베딩 벡터
- **사용처**: Classification, retrieval, fairness analysis 등
- **설명**: 
  - 100개 = DETR의 object queries 개수
  - 256 = hidden dimension (임베딩 차원)
  - 각 query는 하나의 potential object를 나타냄

### 2) decoder_output
```
Shape: [6, 100, 256]
Type: torch.FloatTensor
```
- **의미**: 모든 decoder layer의 hidden states
- **사용처**: Layer-wise 분석, 중간 표현 연구
- **설명**:
  - 6 = decoder layers 개수
  - `decoder_output[-1]` = `query_embeddings` (마지막 레이어)

### 3) encoder_output
```
Shape: [H*W, 256]
Type: torch.FloatTensor
예: [850, 256] or [1600, 256]
```
- **의미**: Transformer encoder의 출력 (전역 context)
- **사용처**: Scene-level 표현, spatial reasoning
- **설명**:
  - H*W = feature map의 spatial 크기 (이미지 크기에 따라 다름)
  - 256 = hidden dimension

### 4) backbone_features
```
Shape: [2048, H, W]
Type: torch.FloatTensor
예: [2048, 25, 34]
```
- **의미**: ResNet backbone의 마지막 feature map
- **사용처**: 낮은 수준의 시각적 특징 분석
- **설명**:
  - 2048 = ResNet-50의 마지막 채널 수
  - H, W = feature map의 spatial 차원

### 5) pred_logits
```
Shape: [100, 92]
Type: torch.FloatTensor
```
- **의미**: 각 query의 class prediction logits
- **사용처**: Detection 결과, confidence 계산
- **설명**:
  - 100 = queries
  - 92 = COCO classes (91) + no-object (1)

### 6) pred_boxes
```
Shape: [100, 4]
Type: torch.FloatTensor
```
- **의미**: 각 query의 bounding box 좌표
- **사용처**: Detection 결과, spatial 분석
- **설명**:
  - 4 = (center_x, center_y, width, height)
  - 값 범위: [0, 1] (normalized)

---

## 💡 5. 파일 읽기 예제

### Python에서 읽기

```python
import torch

# 단일 파일 읽기
embedding = torch.load('embeddings_output/embedding_000000000001.pt')

print("Keys:", embedding.keys())
print("Query embeddings shape:", embedding['query_embeddings'].shape)
print("Predictions shape:", embedding['pred_logits'].shape)

# Query embeddings 추출
query_emb = embedding['query_embeddings']  # [100, 256]
print(f"Shape: {query_emb.shape}")
print(f"Mean: {query_emb.mean():.4f}")
print(f"Std: {query_emb.std():.4f}")

# Detection 결과 확인
logits = embedding['pred_logits']         # [100, 92]
boxes = embedding['pred_boxes']           # [100, 4]

# Person class (COCO class 0)
person_probs = logits.softmax(-1)[:, 0]   # [100]
confident_detections = person_probs > 0.5
print(f"High confidence detections: {confident_detections.sum()}")
```

### Metadata 읽기

```python
import json

with open('embeddings_output/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Total images: {metadata['num_images']}")
print(f"Image IDs: {metadata['image_ids'][:10]}...")  # 처음 10개
print(f"Embedding keys: {metadata['embedding_keys']}")
```

---

## 📏 6. 파일 크기 예상

### 단일 이미지당 크기 (extract_type별)

**decoder (권장)**
- query_embeddings: 100 * 256 * 4 bytes = 100KB
- decoder_output: 6 * 100 * 256 * 4 bytes = 600KB
- encoder_output: ~850 * 256 * 4 bytes = ~850KB
- pred_logits: 100 * 92 * 4 bytes = 37KB
- pred_boxes: 100 * 4 * 4 bytes = 1.6KB
- **총 약 1.6MB/이미지**

**all**
- 위 내용 + backbone_features: 2048 * 25 * 34 * 4 bytes = ~7MB
- **총 약 8-9MB/이미지**

**encoder**
- encoder_output + predictions
- **총 약 1MB/이미지**

**backbone**
- backbone_features + predictions
- **총 약 7-8MB/이미지**

### 전체 데이터셋 크기 예상

| 이미지 수 | decoder | all | encoder | backbone |
|----------|---------|-----|---------|----------|
| 100      | 160MB   | 900MB | 100MB | 800MB |
| 1,000    | 1.6GB   | 9GB   | 1GB   | 8GB   |
| 10,000   | 16GB    | 90GB  | 10GB  | 80GB  |

**권장**: `extract_type='decoder'` 사용 시 가장 적절한 용량/정보 비율

---

## 🎯 7. 실제 사용 예제

### 모든 임베딩을 numpy array로

```python
import torch
import numpy as np
from pathlib import Path

# 모든 query embeddings를 수집
embeddings_list = []
image_ids = []

output_dir = Path('embeddings_output')
for pt_file in sorted(output_dir.glob('embedding_*.pt')):
    emb = torch.load(pt_file)
    embeddings_list.append(emb['query_embeddings'].numpy())  # [100, 256]
    
    # 파일명에서 image_id 추출
    image_id = int(pt_file.stem.split('_')[1])
    image_ids.append(image_id)

# 하나의 큰 행렬로 결합
all_embeddings = np.stack(embeddings_list, axis=0)  # [num_images, 100, 256]
print(f"All embeddings shape: {all_embeddings.shape}")

# 평균 pooling으로 각 이미지를 단일 벡터로
image_vectors = all_embeddings.mean(axis=1)  # [num_images, 256]
print(f"Image vectors shape: {image_vectors.shape}")
```

### 특정 이미지들만 로드

```python
import json
import torch

# Metadata에서 이미지 ID 목록 가져오기
with open('embeddings_output/metadata.json', 'r') as f:
    metadata = json.load(f)

# 처음 10개 이미지만
for image_id in metadata['image_ids'][:10]:
    emb_path = f'embeddings_output/embedding_{image_id:012d}.pt'
    emb = torch.load(emb_path)
    print(f"Image {image_id}: {emb['query_embeddings'].shape}")
```

---

## ✅ 요약

### Output 위치
- **폴더**: `--output_dir`로 지정 (기본값: `./embeddings_output`)
- **파일 개수**: 이미지 수 + 1 (metadata.json)

### 파일 형식
- **metadata.json**: JSON 텍스트 파일
- **embedding_*.pt**: PyTorch 텐서 딕셔너리 (pickle)

### 주요 내용
- **query_embeddings** [100, 256]: 가장 중요! 각 object의 임베딩
- **pred_logits** [100, 92]: Detection class scores
- **pred_boxes** [100, 4]: Bounding box 좌표

### 권장 사용
```bash
# Query embeddings만 추출 (용량 효율적)
python extract_embeddings.py \
    --extract_type decoder \
    --output_dir ./my_embeddings \
    ...
```

### 읽기
```python
import torch
emb = torch.load('embeddings_output/embedding_000000000001.pt')
query_emb = emb['query_embeddings']  # [100, 256]
```
