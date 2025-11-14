# DETR 임베딩 추출 도구

FACET 데이터셋을 DETR 모델에 통과시켜 **임베딩 벡터 f**를 추출하는 도구입니다.

## 📁 위치

모든 임베딩 관련 파일은 `embedding/` 폴더에 있습니다.

```
detr/
├── embedding/              ← 여기!
│   ├── extract_embeddings.py
│   ├── analyze_embeddings.py
│   ├── extract_facet_embeddings.sh
│   ├── test_extraction.sh
│   ├── INDEX.md           ← 파일 가이드
│   ├── QUICKSTART.md      ← 빠른 시작
│   └── README.md          ← 상세 문서
├── models/
├── datasets/
└── main.py
```

## 🚀 빠른 시작

### 1. 테스트 실행
```bash
cd embedding
bash test_extraction.sh
```

### 2. 전체 데이터셋에서 임베딩 추출
```bash
cd embedding

# Python 직접 실행 (main.py 스타일)
python extract_embeddings.py \
    --batch_size 16 \
    --no_aux_loss \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /home/dohyeong/Desktop/FACET\(원본\) \
    --output_dir ./embeddings_output \
    --extract_type decoder

# 또는 Bash 스크립트 사용
bash extract_facet_embeddings.sh --extract_type decoder
```

### 3. 추출된 임베딩 사용
```python
from analyze_embeddings import EmbeddingAnalyzer

analyzer = EmbeddingAnalyzer('./embeddings_output')
matrix, image_ids, _ = analyzer.get_query_embeddings_as_matrix()
print(f"Extracted embeddings: {matrix.shape}")  # [num_images*100, 256]
```

## 📚 문서

- **[embedding/INDEX.md](embedding/INDEX.md)** - 전체 파일 가이드
- **[embedding/QUICKSTART.md](embedding/QUICKSTART.md)** - 빠른 시작 (추천!)
- **[embedding/README.md](embedding/README.md)** - 상세 사용 설명서

## 💡 주요 기능

### 추출 가능한 임베딩
- ✅ **Query Embeddings** - 가장 중요! (각 object의 고수준 임베딩)
- ✅ Encoder Output - 전역 context
- ✅ Backbone Features - 낮은 수준 시각 특징
- ✅ Decoder Output - 모든 레이어의 hidden states

### 지원 기능
- ✅ DETR 공식 pretrained 모델 사용 가능
- ✅ Skin tone별 필터링
- ✅ Lighting 조건별 필터링
- ✅ 배치 처리로 빠른 추출
- ✅ 임베딩 분석 및 시각화 도구

## 🎯 사용 예제

### FACET 데이터셋
```bash
cd embedding
python extract_embeddings.py \
    --batch_size 16 \
    --no_aux_loss \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /home/dohyeong/Desktop/FACET\(원본\) \
    --output_dir ./embeddings_facet \
    --extract_type decoder
```

### COCO 데이터셋
```bash
cd embedding
python extract_embeddings.py \
    --batch_size 16 \
    --no_aux_loss \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /home/dohyeong/Desktop/COCO \
    --output_dir ./embeddings_coco \
    --extract_type decoder
```

### Skin Tone별
```bash
cd embedding
python extract_embeddings.py \
    --batch_size 16 \
    --no_aux_loss \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /home/dohyeong/Desktop/FACET\(원본\) \
    --output_dir ./embeddings_skin1 \
    --extract_type decoder \
    --skin skin1
```

## 📦 출력 형식

```
embeddings_output/
├── metadata.json
├── embedding_000000000001.pt
├── embedding_000000000002.pt
└── ...
```

각 `.pt` 파일:
```python
{
    'query_embeddings': Tensor[100, 256],  # 가장 중요!
    'pred_logits': Tensor[100, num_classes],
    'pred_boxes': Tensor[100, 4],
}
```

## 🔧 문제 해결

### CUDA out of memory
```bash
python extract_embeddings.py --batch_size 8 ...
```

### 빠른 테스트
```bash
cd embedding
bash test_extraction.sh
```

## 📖 더 알아보기

자세한 내용은 `embedding/` 폴더의 문서를 참고하세요:
- 시작: `embedding/QUICKSTART.md`
- 파일 가이드: `embedding/INDEX.md`
- 상세 설명: `embedding/README.md`

---

**지금 시작하기**: `cd embedding && bash test_extraction.sh`
