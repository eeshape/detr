# DETR 임베딩 추출 도구 - 파일 가이드

이 폴더에는 DETR 모델에서 임베딩 벡터 **f**를 추출하는 모든 도구가 포함되어 있습니다.

## 📚 문서

### 1. [QUICKSTART.md](QUICKSTART.md) ⭐ **먼저 읽어보세요!**
- 빠른 시작 가이드
- 즉시 사용 가능한 명령어 예제
- 일반적인 사용 케이스

### 2. [README.md](README.md)
- 상세한 사용 설명서
- 모든 기능과 옵션 설명
- 고급 활용 방법

## 🛠️ 실행 스크립트

### 메인 스크립트

**`extract_embeddings.py`** - 임베딩 추출 메인 프로그램
- DETR 모델에서 다양한 임베딩 추출
- main.py와 동일한 인터페이스
- 사용 예제:
  ```bash
  python extract_embeddings.py \
      --batch_size 16 \
      --no_aux_loss \
      --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
      --coco_path /home/dohyeong/Desktop/COCO \
      --output_dir ./embeddings_output \
      --extract_type decoder
  ```

### 편의 스크립트

**`extract_facet_embeddings.sh`** - FACET 데이터셋용 실행 스크립트
- 기본 설정이 미리 구성됨
- 사용법:
  ```bash
  bash extract_facet_embeddings.sh --extract_type decoder
  ```

**`test_extraction.sh`** - 테스트용 스크립트
- 소량 데이터로 빠른 테스트
- 전체 실행 전 권장
- 사용법:
  ```bash
  bash test_extraction.sh
  ```

## 🔍 분석 도구

**`analyze_embeddings.py`** - 추출된 임베딩 분석 및 시각화
- 임베딩 로드 및 통계 분석
- PCA/t-SNE 시각화
- 사용 예제:
  ```python
  from analyze_embeddings import EmbeddingAnalyzer
  analyzer = EmbeddingAnalyzer('./embeddings_output')
  matrix, ids, _ = analyzer.get_query_embeddings_as_matrix()
  ```

**`demo_extract_single_image.py`** - 단일 이미지 임베딩 추출 데모
- 개별 이미지 처리 예제
- 임베딩 추출 과정 학습용

## 🎯 추출되는 임베딩 타입

### 1. Query Embeddings ⭐ **가장 중요!**
- Shape: `[num_queries, hidden_dim]` (기본: [100, 256])
- DETR decoder의 최종 출력
- 각 object의 고수준 semantic 임베딩
- **대부분의 downstream task에 가장 유용**

### 2. Encoder Output
- Shape: `[H*W, hidden_dim]`
- Transformer encoder 출력
- 전역 context 정보

### 3. Backbone Features
- Shape: `[C, H, W]`
- ResNet 등 backbone의 feature map
- 낮은 수준의 시각적 특징

### 4. Decoder Output
- Shape: `[num_layers, num_queries, hidden_dim]`
- 모든 decoder layer의 hidden states

## 🚀 빠른 실행 체크리스트

1. ✅ **테스트 실행**
   ```bash
   cd /home/dohyeong/Desktop/detr/embedding
   bash test_extraction.sh
   ```

2. ✅ **전체 데이터셋 처리**
   ```bash
   python extract_embeddings.py \
       --batch_size 16 \
       --no_aux_loss \
       --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
       --coco_path /home/dohyeong/Desktop/FACET\(원본\) \
       --output_dir ./embeddings_output \
       --extract_type decoder
   ```

3. ✅ **결과 확인**
   ```python
   from analyze_embeddings import EmbeddingAnalyzer
   analyzer = EmbeddingAnalyzer('./embeddings_output')
   print(analyzer.metadata)
   ```

## 📦 출력 구조

```
embeddings_output/
├── metadata.json              # 메타데이터 (이미지 ID 목록 등)
├── embedding_000000000001.pt  # 이미지 ID 1의 임베딩
├── embedding_000000000002.pt  # 이미지 ID 2의 임베딩
└── ...
```

각 `.pt` 파일:
```python
{
    'query_embeddings': Tensor[100, 256],  # 가장 중요!
    'pred_logits': Tensor[100, num_classes],
    'pred_boxes': Tensor[100, 4],
    # 기타 (extract_type에 따라)
}
```

## ❓ 도움말

### 어떤 임베딩을 추출해야 하나요?
- **일반적인 경우**: `--extract_type decoder` (Query embeddings)
- **연구/분석용**: `--extract_type all`

### GPU 메모리 부족
```bash
# batch_size 줄이기
python extract_embeddings.py --batch_size 8 ...
```

### 체크포인트 문제
- DETR 공식 pretrained 사용 권장:
  ```
  https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
  ```

## 📞 주요 명령어 요약

| 작업 | 명령어 |
|------|--------|
| 테스트 | `bash test_extraction.sh` |
| FACET 전체 | `bash extract_facet_embeddings.sh` |
| COCO | `python extract_embeddings.py --coco_path /path/to/COCO ...` |
| 분석 | `python -c "from analyze_embeddings import example_usage; example_usage()"` |

---

**시작하기**: [QUICKSTART.md](QUICKSTART.md)를 읽고 `bash test_extraction.sh`를 실행하세요!
